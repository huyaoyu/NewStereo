
from __future__ import print_function

import numpy as np
import time

from workflow import WorkFlow, TorchFlow

import ArgumentParser

from TTPWCNS import TrainTestPWCNetStereo

def print_delimeter(c = "=", n = 20, title = "", leading = "\n", ending = "\n"):
    d = [c for i in range( int(n/2) )]

    if ( 0 == len(title) ):
        s = "".join(d) + "".join(d)
    else:
        s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

# Template for custom WorkFlow object.
class MyWF(TorchFlow.TorchFlow):
    def __init__(self, workingDir, prefix = "", suffix = "", disableStreamLogger=False):
        super(MyWF, self).__init__(workingDir, prefix, suffix, disableStreamLogger)

        # === Custom member variables. ===
        self.tt = None # The TrainTestBase object.

    def set_tt(self, tt):
        self.tt = tt

    def check_tt(self):
        if ( self.tt is None ):
            Exception("self.tt must not be None.")

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        self.check_tt()

        # === Custom code. ===

        self.tt.initialize()

        self.logger.info("Initialized.")

        self.post_initialize()

    # Overload the function train().
    def train(self, imgL, imgR, dispL, gradL, gradR, epochCount):
        super(MyWF, self).train()

        self.check_tt()

        return self.tt.train(imgL, imgR, dispL, gradL, gradR, epochCount)
        
    # Overload the function test().
    def test(self, imgL, imgR, dispL, gradL, gradR, epochCount, flagSave=True):
        super(MyWF, self).test()

        self.check_tt()

        return self.tt.test(imgL, imgR, dispL, gradL, gradR, epochCount, flagSave)

    def infer(self, imgL, imgR, gradL, Q):

        self.check_tt()

        self.tt.infer( imgL, imgR, gradL, Q )

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        self.check_tt()

        self.tt.finalize()

        self.logger.info("Finalized.")

if __name__ == "__main__":
    print("Hello NG.")

    # Handle the arguments.
    args = ArgumentParser.args

    # Handle the crop settings.
    cropTrain = ArgumentParser.convert_str_2_int_list( args.dl_crop_train )
    cropTest  = ArgumentParser.convert_str_2_int_list( args.dl_crop_test )

    # Handle the resize settings.
    newSize = ArgumentParser.convert_str_2_int_list( args.dl_resize )

    print_delimeter(title = "Before WorkFlow initialization." )

    try:
        # Instantiate an object for MyWF.
        wf = MyWF(args.working_dir, prefix=args.prefix, suffix=args.suffix, disableStreamLogger=False)
        wf.verbose = False

        # Cross reference.
        tt = TrainTestPWCNetStereo(wf.workingDir, wf)
        wf.set_tt(tt)

        if ( True == args.multi_gpus ):
            tt.enable_multi_GPUs()

        # Set parameters.
        tt.set_optimizer_type(args.optimizer)
        tt.set_learning_rate(args.lr)
        tt.set_data_loader_params( \
            args.dl_batch_size, not args.dl_disable_shuffle, args.dl_num_workers, args.dl_drop_last, \
            cropTrain=cropTrain, cropTest=cropTest, newSize=newSize )
        
        if ( args.grayscale ):
            tt.enable_grayscale()
        
        if ( args.sobel_x ):
            tt.enable_Sobel_x()

        tt.set_dataset_root_dir( args.data_root_dir, args.data_entries, args.data_file_list )
        tt.set_read_model( args.read_model )
        tt.set_read_optimizer( args.read_optimizer )
        tt.enable_auto_save( args.auto_save_model )
        tt.set_training_acc_params( args.train_interval_acc_write, args.train_interval_acc_plot, args.use_intermittent_plotter )

        if ( True == args.test ):
            tt.switch_on_test()
        else:
            tt.switch_off_test()

        if ( True == args.infer ):
            tt.switch_on_infer()
        else:
            tt.switch_off_infer()

        if ( True == args.inspect ):
            tt.flagInspect = True
            print("Inspection enabled.")
        else:
            tt.flagInspect = False

        tt.set_max_disparity(args.max_disparity)
        tt.set_corr_kernel_size(args.corr_k)
        tt.set_flow_amp(args.flow_amp)

        # Initialization.
        print_delimeter(title = "Initialize.")
        wf.initialize()

        if ( False == args.infer ):
            # Get the number of test data.
            nTests = len( tt.imgTestLoader )
            wf.logger.info("The size of the test dataset is %s." % ( nTests ))

            if ( False == args.test ):
                # Create the test data iterator.
                iterTestData = iter( tt.imgTestLoader )

                # Training loop.
                wf.logger.info("Begin training.")
                print_delimeter(title = "Training loops.")

                testImgL, testImgR, testDispL, testGradL, testGradR = next( iterTestData )
                wf.test( testImgL, testImgR, testDispL, testGradL, testGradR, 0 )

                for i in range(args.train_epochs):
                    for batchIdx, ( imgL, imgR, dispL, gradL, gradR ) in enumerate( tt.imgTrainLoader ):
                        
                        wf.train( imgL, imgR, dispL, gradL, gradR, i )

                        if ( True == tt.flagInspect ):
                            wf.logger.warning("Inspection enabled.")

                        # Check if we need a test.
                        if ( 0 != args.test_loops ):
                            if ( tt.countTrain % args.test_loops == 0 ):
                                # Get test data.
                                try:
                                    testImgL, testImgR, testDispL, testGradL, testGradR = next( iterTestData )
                                except StopIteration:
                                    iterTestData = iter(tt.imgTestLoader)
                                    testImgL, testImgR, testDispL, testGradL, testGradR = next( iterTestData )

                                # Perform test.
                                wf.test( testImgL, testImgR, testDispL, testGradL, testGradR, i )
            else:
                wf.logger.info("Begin testing.")
                print_delimeter(title="Testing loops.")

                testLossList = []

                for batchIdx, ( imgL, imgR, dispL, gradL, gradR ) in enumerate( tt.imgTestLoader ):
                    loss, metrics = wf.test( imgL, imgR, dispL, gradL, gradR, batchIdx, args.test_flag_save )

                    if ( True == tt.flagInspect ):
                        wf.logger.warning("Inspection enabled.")

                    wf.logger.info("Test %d, lossTest = %f." % ( batchIdx, loss ))

                    testLossList.append( [ imgL.size()[0], loss, *(np.mean(metrics, axis=0).tolist()) ] )

                    if ( args.test_loops > 0 and batchIdx >= args.test_loops - 1 ):
                        break

                # import ipdb; ipdb.set_trace()

                testLossAndMetrics = np.array(testLossList, dtype=np.float32)
                scaledLossAndMetrics = testLossAndMetrics[:, 1:] * testLossAndMetrics[:, 0].reshape((-1,1))
                averagedLossAndMetrics = np.mean(scaledLossAndMetrics, axis=0)

                wf.logger.info("Average loss = %f." % ( averagedLossAndMetrics[0] ))
                wf.logger.info("Average 1-pixel error rate = %f." % ( averagedLossAndMetrics[1] ))
                wf.logger.info("Average 2-pixel error rate = %f." % ( averagedLossAndMetrics[2] ))
                wf.logger.info("Average 3-pixel error rate = %f." % ( averagedLossAndMetrics[3] ))
                wf.logger.info("Average 4-pixel error rate = %f." % ( averagedLossAndMetrics[4] ))
                wf.logger.info("Average 5-pixel error rate = %f." % ( averagedLossAndMetrics[5] ))
                wf.logger.info("Average end point error = %f." % ( averagedLossAndMetrics[6] ))

                # Save the loss values to file the working directory.
                testResultSummaryFn = wf.compose_file_name("BatchTest", "dat", subFolder=tt.testResultSubfolder)
                np.savetxt( testResultSummaryFn, testLossList)
        else:
            wf.logger.info("Begin inferring.")
            print_delimeter(title="Inferring loops.")

            # for batchIdx, ( imgL, imgR, Q ) in enumerate( tt.imgInferLoader ):
            #     startT = time.time()
            #     wf.infer( imgL, imgR, Q)
            #     endT = time.time()
            #     wf.logger.info("Infer time: %fs." % (endT - startT))

            #     if ( True == tt.flagInspect ):
            #         wf.logger.warning("Inspection enabled")

            #     wf.logger.info("Infer %d." % ( batchIdx ))

            #     if ( tt.countTest == args.test_loops ):
            #         wf.logger.info("Infer reaches the maximum number. Maximum is %d. " % (args.test_loops))
            #         break

            wf.logger.info("Done inferring.")

        wf.finalize()
    except WorkFlow.SigIntException as sie:
        print("SigInt revieved, perform finalize...")
        wf.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")
