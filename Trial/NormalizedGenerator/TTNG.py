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

from CommonPython.PointCloud.PLYHelper import write_PLY

import matplotlib.pyplot as plt
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("TTNG: Environment variable DISPLAY is not present in the system.")
    print("TTNG: Switch the backend of matplotlib to agg.")

Q_FLIP = np.array( [ \
    [ 1.0,  0.0,  0.0, 0.0 ], \
    [ 0.0, -1.0,  0.0, 0.0 ], \
    [ 0.0,  0.0, -1.0, 0.0 ], \
    [ 0.0,  0.0,  0.0, 1.0 ] ], dtype=np.float32 )

class TrainTestNG(TrainTestBase):
    def __init__(self, workingDir, frame=None):
        super(TrainTestNG, self).__init__( workingDir, frame )

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
        
        # self.model = 

        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                raise Exception("Model file (%s) does not exist." % ( modelFn ))

            self.model = self.frame.load_model( self.model, modelFn )

        if ( self.flagCPU ):
            self.model.set_cpu_mode()

        # self.frame.logger.info("NG has %d model parameters." % \
        #     ( sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )
    
    # def post_init_model(self):
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_optimizer(self):
        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        # self.optimizer = optim.Adam( self.model.parameters(), lr=self.learningRate )
        self.frame.logger.warning("init_optimizer() has no effect for dummy test.")

    # Overload parent's function.
    def train(self, imgL, imgR, dispL, dispLH, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not train with infer mode.")

        # self.model.train()
        imgL   = Variable( torch.FloatTensor(imgL) )
        imgR   = Variable( torch.FloatTensor(imgR) )
        dispL  = Variable( torch.FloatTensor(dispL) )
        dispLH = Variable( torch.FloatTensor(dispLH) )

        if ( not self.flagCPU ):
            imgL   = imgL.cuda()
            imgR   = imgR.cuda()
            dispL  = dispL.cuda()
            dispLH = dispLH.cuda()

        # self.optimizer.zero_grad()

        # loss = F.smooth_l1_loss()

        # loss.backward()

        # self.optimizer.step()

        # self.frame.AV["loss"].push_back( loss.item() )
        self.frame.AV["loss"].push_back( self.countTrain )

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
                self.frame.logger.info("Auto-save the model.")
                self.frame.save_model( self.model, modelName )

        self.frame.logger.info("E%d, L%d: %s" % (epochCount, self.countTrain, self.frame.get_log_str()))

    # Overload parent's function.
    def test(self, imgL, imgR, dispL, dispLH, epochCount):
        self.check_frame()

        if ( True == self.flagInfer ):
            raise Exception("Could not test in the infer mode.")

        # self.model.eval()
        imgL = Variable( torch.FloatTensor( imgL ) )
        imgR = Variable( torch.FloatTensor( imgR ) )

        if ( not self.flagCPU ):
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        with torch.no_grad():
            # Forward.
            pass

        # loss

        self.countTest += 1

        if ( True == self.flagTest ):
            count = self.countTest
        else:
            count = self.countTrain

        # Test the existance of an AccumulatedValue object.
        if ( True == self.frame.have_accumulated_value("lossTest") ):
            # self.frame.AV["lossTest"].push_back(loss.item(), self.countTest)
            self.frame.AV["lossTest"].push_back(count, self.countTest)
        else:
            self.frame.logger.info("Could not find \"lossTest\"")

        self.frame.plot_accumulated_values()

        return loss.item()

    def infer(self, imgL, imgR, dispLH, Q):
        self.check_frame()

        self.model.eval()
        imgL   = Variable( torch.FloatTensor( imgL ) )
        imgR   = Variable( torch.FloatTensor( imgR ) )
        dispLH = Variable( torch.FloatTensor( dispLH ) )

        if ( not self.flagCPU ):
            imgL   = imgL.cuda()
            imgR   = imgR.cuda()
            dispLH = dispLH.cuda()

        with torch.no_grad():
            # Forward.
            pass

        self.countTest += 1

    # Overload parent's function.
    def finalize(self):
        self.check_frame()

        # # Save the model.
        # if ( False == self.flagTest and False == self.flagInfer ):
        #     self.frame.save_model( self.model, "NG" )
        self.frame.logger.warning("Model not saved for dummy test.")
