from __future__ import print_function

import copy
import cv2
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from workflow import WorkFlow, TorchFlow

from TTBase import TrainTestBase
from Model.SmallSizeStereo import SmallSizeStereo, SmallSizeStereoParams

from CommonPython.PointCloud.PLYHelper import write_PLY

import matplotlib.pyplot as plt
# if ( not ( "DISPLAY" in os.environ ) ):
#     plt.switch_backend('agg')
#     print("TTNG: Environment variable DISPLAY is not present in the system.")
#     print("TTNG: Switch the backend of matplotlib to agg.")
plt.switch_backend('agg')
print("TTNG: Switch the backend of matplotlib to agg.")

Q_FLIP = np.array( [ \
    [ 1.0,  0.0,  0.0, 0.0 ], \
    [ 0.0, -1.0,  0.0, 0.0 ], \
    [ 0.0,  0.0, -1.0, 0.0 ], \
    [ 0.0,  0.0,  0.0, 1.0 ] ], dtype=np.float32 )

class TrainTestSSS(TrainTestBase):
    def __init__(self, workingDir, frame=None):
        super(TrainTestSSS, self).__init__( workingDir, frame )

    # def initialize(self):
    #     self.check_frame()
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_workflow(self):
        # === Create the AccumulatedObjects. ===
        self.frame.add_accumulated_value("lossTest", 10)

        self.frame.AV["loss"].avgWidth = 10
        
        # ======= AVP. ======
        # === Create a AccumulatedValuePlotter object for ploting. ===
        if ( True == self.flagUseIntPlotter ):
            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "loss", self.frame.AV, ["loss"], [True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "lossTest", self.frame.AV, ["lossTest"], [True], semiLog=True) )
        else:
            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "loss", self.frame.AV, ["loss"], [True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "lossTest", self.frame.AV, ["lossTest"], [True], semiLog=True) )

    # def init_workflow(self):
    #     raise Exception("Not implemented.")

    # def init_torch(self):
    #     raise Exception("Not implemented.")

    # def init_data(self):
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_model(self):
        
        self.params = SmallSizeStereoParams()
        self.model = SmallSizeStereo(self.params)

        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                raise Exception("Model file (%s) does not exist." % ( modelFn ))

            self.model = self.frame.load_model( self.model, modelFn )

        if ( self.flagCPU ):
            self.model.set_cpu_mode()

        self.frame.logger.info("NG has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )
    
    # def post_init_model(self):
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_optimizer(self):
        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        if ( "adam" == self.optType ):
            self.optimizer = optim.Adam( self.model.parameters(), lr=self.learningRate )
        elif ( "sgd" == self.optType ):
            self.optimizer = optim.SGD( self.model.parameters(), lr=self.learningRate )
        else:
            raise Exception("Unexpected optimizer type: {}. ".format(self.optType))

        # Check if we have to read the optimizer state from the filesystem.
        if ( "" != self.readOptimizerString ):
            optFn = "%s/models/%s" % ( self.frame.workingDir, self.readOptimizerString )

            if ( not os.path.isfile( optFn ) ):
                raise Exception("Optimizer file (%s) does not exist. " % ( optFn ))

            self.optimizer = self.frame.load_optimizer(self.optimizer, optFn)

            self.frame.logger.info("Optimizer state loaded for file %s. " % (optFn))

    # Overload parent's function.
    def train(self, imgL, imgR, dispL, gradL, gradR, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not train with infer mode.")

        self.model.train()

        if ( not self.flagCPU ):
            imgL  = imgL.cuda()
            imgR  = imgR.cuda()
            dispL = dispL.cuda()
            gradL = gradL.cuda()
            gradR = gradR.cuda()

        self.optimizer.zero_grad()

        # Forward.
        pred0 = self.model(imgL, imgR, gradL, gradR)

        loss = F.smooth_l1_loss( pred0, dispL, reduction="mean" )
        # loss = F.l1_loss( pred0, dispL, reduction="mean" )
        # loss = F.mse_loss( pred0, dispL ) + \
        #        F.mse_loss( pred1, dispL )

        loss.backward()

        self.optimizer.step()

        self.frame.AV["loss"].push_back( loss.item() )

        self.countTrain += 1

        if ( self.countTrain % self.trainIntervalAccWrite == 0 ):
            self.frame.write_accumulated_values()

        # Plot accumulated values.
        if ( self.countTrain % self.trainIntervalAccPlot == 0 ):
            self.frame.plot_accumulated_values()

        # Auto-save.
        if ( 0 != self.autoSaveModelLoops ):
            if ( self.countTrain % self.autoSaveModelLoops == 0 ):
                modelName = "AutoSave_%08d" % ( self.countTrain )
                optName   = "AutoSave_Opt_%08d" % ( self.countTrain )
                self.frame.logger.info("Auto-save the model and optimizer.")
                self.frame.save_model( self.model, modelName )
                self.frame.save_optimizer( self.optimizer, optName )

        self.frame.logger.info("E%d, L%d: %s" % (epochCount, self.countTrain, self.frame.get_log_str()))

    def draw_test_results(self, identifier, predD, trueD, imgL, imgR):
        """
        Draw test results.

        predD: Dimension (B, 1, H, W).
        trueD: Dimension (B, 1, H, W).
        imgL: Dimension (B, C, H, W).
        imgR: Dimension (B, C, H, W).
        """

        batchSize = predD.size()[0]
        
        for i in range(batchSize):
            outDisp = predD[i, 0, :, :].detach().cpu().numpy()
            gdtDisp = trueD[i, 0, :, :].detach().cpu().numpy()
            # import ipdb; ipdb.set_trace()

            gdtMin = gdtDisp.min()
            gdtMax = gdtDisp.max()

            # outDisp = outDisp - outDisp.min()
            outDisp = outDisp - gdtMin
            gdtDisp = gdtDisp - gdtMin

            # outDisp = outDisp / outDisp.max()
            outDisp = np.clip( outDisp / gdtMax, 0.0, 1.0 )
            gdtDisp = gdtDisp / gdtMax

            # Create a matplotlib figure.
            fig = plt.figure(figsize=(12.8, 9.6), dpi=300)

            ax = plt.subplot(2, 2, 1)
            plt.tight_layout()
            ax.set_title("Ref")
            ax.axis("off")
            img0 = imgL[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img0 = img0 - img0.min()
            img0 = img0 / img0.max()
            if ( 1 == img0.shape[2] ):
                img0 = img0[:, :, 0]
            plt.imshow( img0 )

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title("Tst")
            ax.axis("off")
            img1 = imgR[i, :, :, :].permute((1,2,0)).cpu().numpy()
            img1 = img1 - img1.min()
            img1 = img1 / img1.max()
            if ( 1 == img1.shape[2] ):
                img1 = img1[:, :, 0]
            plt.imshow( img1 )

            ax = plt.subplot(2, 2, 2)
            plt.tight_layout()
            ax.set_title("Ground truth")
            ax.axis("off")
            plt.imshow( gdtDisp )

            ax = plt.subplot(2, 2, 4)
            plt.tight_layout()
            ax.set_title("Prediction")
            ax.axis("off")
            plt.imshow( outDisp )

            figName = "%s_%02d" % (identifier, i)
            figName = self.frame.compose_file_name(figName, "png", subFolder=self.testResultSubfolder)
            plt.savefig(figName)

            plt.close(fig)

    def save_test_disp(self, identifier, pred):
        batchSize = pred.size()[0]
        
        for i in range(batchSize):
            # Get the prediction.
            disp = pred[i, 0, :, :].cpu().numpy()

            fn = "%s_%02d" % (identifier, i)
            fn = self.frame.compose_file_name(fn, "dat", subFolder=self.testResultSubfolder)

            np.savetxt( fn, disp, fmt="%+.6f" )

    # Overload parent's function.
    def test(self, imgL, imgR, dispL, gradL, gradR, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not test in the infer mode.")

        self.model.eval()

        if ( not self.flagCPU ):
            imgL  = imgL.cuda()
            imgR  = imgR.cuda()
            dispL = dispL.cuda()
            gradL = gradL.cuda()
            gradR = gradR.cuda()

        # import ipdb; ipdb.set_trace()

        with torch.no_grad():
            # Forward.
            pred0 = self.model( imgL, imgR, gradL, gradR )
            loss = F.smooth_l1_loss( pred0, dispL, reduction="mean" )
            # loss = torch.mean( torch.abs( pred0 - dispL ) )

        self.countTest += 1

        if ( True == self.flagTest ):
            count = self.countTest
        else:
            count = self.countTrain

        # Draw and save results.
        identifier = "test_%d" % (count - 1)
        self.draw_test_results( identifier, pred0, dispL, dispLH, imgL )
        # self.draw_test_results( identifier, pred0, dispL, pred0, imgL )
        self.save_test_disp( identifier, pred0 )

        # Test the existance of an AccumulatedValue object.
        if ( True == self.frame.have_accumulated_value("lossTest") ):
            self.frame.AV["lossTest"].push_back(loss.item(), self.countTest)
        else:
            self.frame.logger.info("Could not find \"lossTest\"")

        self.frame.plot_accumulated_values()

        return loss.item()

    def infer(self, imgL, imgR, gradL, gradR, Q):
        self.check_frame()

        self.model.eval()

        if ( not self.flagCPU ):
            imgL  = imgL.cuda()
            imgR  = imgR.cuda()
            gradL = gradL.cuda()
            gradR = gradR.cuda()

        with torch.no_grad():
            # Forward.
            pass

        self.countTest += 1

    # Overload parent's function.
    def finalize(self):
        self.check_frame()

        # Save the model and optimizer.
        if ( False == self.flagTest and False == self.flagInfer ):
            self.frame.save_model( self.model, "SSS" )
            self.frame.save_optimizer( self.optimizer, "SSS_Opt" )
        # self.frame.logger.warning("Model not saved for dummy test.")
