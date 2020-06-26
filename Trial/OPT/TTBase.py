from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from workflow import WorkFlow, TorchFlow

from DataLoader.SceneFlow import Loader as DA
from DataLoader import PreProcess
from DataLoader.SceneFlow.utils import list_files_sceneflow_FlyingThings, read_string_list_2D

RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL = 100

# DATASET_LIST_TRAINING_IMG_L = "InputImageTrainL.txt"
# DATASET_LIST_TRAINING_IMG_R = "InputImageTrainR.txt"
# DATASET_LIST_TRAINING_DSP_L = "InputDisparityTrain.txt"

# DATASET_LIST_TESTING_IMG_L = "InputImageTestL.txt"
# DATASET_LIST_TESTING_IMG_R = "InputImageTestR.txt"
# DATASET_LIST_TESTING_DSP_L = "InputDisparityTest.txt"

# DATASET_LIST_INFERRING_IMG_L = "InputImageInferL.txt"
# DATASET_LIST_INFERRING_IMG_R = "InputImageInferR.txt"
# DATASET_LIST_INFERRING_Q     = "InputQ.txt"

DATASET_LIST_TRAINING  = "Train.txt"
DATASET_LIST_TESTING   = "Test.txt"
DATASET_LIST_INFERRING = "Infer.txt"

class TrainTestBase(object):
    def __init__(self, workingDir, frame=None):
        self.wd = workingDir
        self.frame = frame

        # NN.
        self.countTrain = 0
        self.countTest  = 0

        self.flagGrayscale = False
        self.flagSobelX    = False

        self.trainIntervalAccWrite = 10    # The interval to write the accumulated values.
        self.trainIntervalAccPlot  = 1     # The interval to plot the accumulate values.
        self.flagUseIntPlotter     = False # The flag of intermittent plotter.

        self.imgTrainLoader  = None
        self.imgTestLoader   = None
        self.imgInferLoader  = None
        self.datasetRootDir  = "./"
        self.dataFileList    = False # Once set to be true, the files contain the list of input dataset will be used.
        self.dataFileListDir = None # If self.dataFileList is True, then this variable must be set properly.
        self.dataEntries     = 0 # 0 for using all the data.
        self.datasetTrain    = None # Should be an object of torch.utils.data.Dataset.
        self.datasetTest     = None # Should be an object of torch.utils.data.Dataset.
        self.datasetInfer    = None # Should be an object of torch.utils.data.Dataset.
        self.dlBatchSize     = 2
        self.dlShuffle       = True
        self.dlNumWorkers    = 2
        self.dlDropLast      = False
        self.dlResize        = (0, 0) # (0, 0) for disable.
        self.dlCropTrain     = (0, 0) # (0, 0) for disable.
        self.dlCropTest      = (0, 0) # (0, 0) for disable.

        self.maxDisparity = 64

        self.model     = None
        self.flagCPU   = False
        self.multiGPUs = False

        self.readModelString     = ""
        self.readOptimizerString = ""
        self.autoSaveModelLoops  = 0 # The number of loops to perform an auto-saving of the model. 0 for disable.

        self.optType   = "adam" # The optimizer type. adam, sgd.
        self.optimizer = None
        self.learningRate = 0.001

        self.testResultSubfolder = "TestResults"

        self.flagTest  = False # Should be set to True when testing.
        self.flagInfer = False # Should be set to True when inferring.

    def initialize(self):
        self.check_frame()

        # Over load these functions if nessesary.
        self.init_base()
        self.init_workflow()
        self.init_torch()
        self.init_data()
        self.init_model()
        self.post_init_model()
        self.init_optimizer()
    
    def train(self):
        self.check_frame()
    
    def test(self):
        self.check_frame()
    
    def finialize(self):
        self.check_frame()

    def infer(self):
        self.check_frame()

    def set_frame(self, frame):
        self.frame = frame
    
    def check_frame(self):
        if ( self.frame is None ):
            raise Exception("self.frame must not be None.")
    
    def enable_grayscale(self):
        self.check_frame()

        self.flagGrayscale = True

        self.frame.logger.info("Grayscale image enabled.")

    def enable_Sobel_x(self):
        self.check_frame()

        self.flagSobelX    = True
        self.flagGrayscale = True

        self.frame.logger.info("Sobel-x image enabled. Image will be converted into grayscale first.")

    def set_learning_rate(self, lr):
        self.check_frame()

        self.learningRate = lr

        if ( self.learningRate >= 1.0 ):
            self.frame.logger.warning("Large learning rate (%f) is set." % (self.learningRate))

    def enable_multi_GPUs(self):
        self.check_frame()

        self.flagCPU   = False
        self.multiGPUs = True

        self.frame.logger.info("Enable multi-GPUs.")

    def set_cpu_mode(self):
        self.check_frame()

        self.flagCPU   = True
        self.multiGPUs = False

        self.frame.logger.warning("CPU mode is selected.")

    def unset_cpu_mode(self):
        self.check_frame()

        self.flagCPU   = False
        self.multiGPUs = False

        self.frame.logger.warning("Back to GPU mode.")

    def set_dataset_root_dir(self, d, nEntries=0, flagFileList=False, fileListDir=None):
        self.check_frame()

        if ( False == os.path.isdir(d) ):
            raise Exception("Dataset directory (%s) not exists." % (d))
        
        self.datasetRootDir = d
        self.dataEntries    = nEntries

        self.frame.logger.info("Data root directory is %s." % ( self.datasetRootDir ))
        if ( 0 != nEntries ):
            self.frame.logger.warning("Only %d entries of the training dataset will be used." % ( nEntries ))

        self.dataFileList    = flagFileList
        self.dataFileListDir = fileListDir

        if ( self.dataFileList ):
            if ( self.dataFileListDir is None ):
                raise Exception("The directory of the file-lists files must be set.")

            if ( not os.path.isdir( self.dataFileListDir ) ):
                raise Exception("File-lists directory %s does not exist. " % ( self.dataFileListDir ))

            self.frame.logger.info("Data loader will use the pre-defined files to load the input data.")
            self.frame.logger.info("The file-list files are saved at %s. " % ( self.dataFileListDir ))

    def set_data_loader_params(self, batchSize=2, shuffle=True, numWorkers=2, dropLast=False, \
        cropTrain=(0, 0), cropTest=(0, 0), newSize=(0, 0)):
        
        self.check_frame()

        self.dlBatchSize  = batchSize
        self.dlShuffle    = shuffle
        self.dlNumWorkers = numWorkers
        self.dlDropLast   = dropLast
        self.dlCropTrain  = cropTrain
        self.dlCropTest   = cropTest
        self.dlResize     = newSize

    def set_read_model(self, readModelString):
        self.check_frame()
        
        self.readModelString = readModelString

        if ( "" != self.readModelString ):
            self.frame.logger.info("Read model from %s." % ( self.readModelString ))

    def set_read_optimizer(self, readOptimizerString):
        self.check_frame()

        self.readOptimizerString = readOptimizerString

        if ( "" != self.readOptimizerString ):
            self.frame.logger.info("Read optimizer from %s. " % ( self.readOptimizerString ))

    def enable_auto_save(self, loops):
        self.check_frame()
        
        self.autoSaveModelLoops = loops

        if ( 0 != self.autoSaveModelLoops ):
            self.frame.logger.info("Auto save enabled with loops = %d." % (self.autoSaveModelLoops))

    def set_training_acc_params(self, intervalWrite, intervalPlot, flagInt=False):
        self.check_frame()
        
        self.trainIntervalAccWrite = intervalWrite
        self.trainIntervalAccPlot  = intervalPlot
        self.flagUseIntPlotter     = flagInt

        if ( True == self.flagUseIntPlotter ):
            if ( self.trainIntervalAccPlot <= RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL ):
                self.frame.logger.warning("When using the intermittent plotter. It is recommended that the plotting interval (%s) is higher than %d." % \
                    ( self.trainIntervalAccPlot, RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL ) )

    def switch_on_test(self):
        self.flagTest = True

    def switch_off_test(self):
        self.flagTest = False

    def switch_on_infer(self):
        self.flagInfer = True

    def switch_off_infer(self):
        self.flagInfer = False

    def set_max_disparity(self, md):
        assert md >= 1
        self.maxDisparity = int(md)

    def init_base(self):
        # Make the subfolder for the test results.
        self.frame.make_subfolder(self.testResultSubfolder)

    def init_workflow(self):
        raise Exception("init_workflow() virtual interface.")

    def init_torch(self):
        self.check_frame()

        self.frame.logger.info("Configure Torch.")

        # torch.manual_seed(1)
        # torch.cuda.manual_seed(1)

    def init_data(self):
        # Get all the sample images.
        # imgTrainL, imgTrainR, dispTrain, imgTestL, imgTestR, dispTest \
        #     = list_files_sample("/home/yyhu/expansion/OriginalData/SceneFlow/Sampler/FlyingThings3D")
        # imgTrainL, imgTrainR, dispTrain, imgTestL, imgTestR, dispTest \
        #     = list_files_sample("/media/yaoyu/DiskE/SceneFlow/Sampler/FlyingThings3D")

        if ( not self.flagInfer ):
            if ( self.dataFileList ):
                imgTrainL, imgTrainR, dispTrain = read_string_list_2D( \
                    self.dataFileListDir + "/" + DATASET_LIST_TRAINING, 3, delimiter=",", prefix=self.datasetRootDir )

                imgTestL, imgTestR, dispTest = read_string_list_2D( \
                    self.dataFileListDir + "/" + DATASET_LIST_TESTING,  3, delimiter=",", prefix=self.datasetRootDir )
            else:
                # imgTrainL, imgTrainR, dispTrain, imgTestL, imgTestR, dispTest \
                #     = list_files_sceneflow_FlyingThings( self.datasetRootDir )
                raise Exception("Listing files on-the-fly not supported.")

            if ( 0 != self.dataEntries ):
                imgTrainL = imgTrainL[0:self.dataEntries]
                imgTrainR = imgTrainR[0:self.dataEntries]
                dispTrain = dispTrain[0:self.dataEntries]
                imgTestL  = imgTestL[0:self.dataEntries]
                imgTestR  = imgTestR[0:self.dataEntries]
                dispTest  = dispTest[0:self.dataEntries]
        else:
            if ( self.dataFileList ):
                imgInferL, imgInferR, Q = read_string_list_2D( \
                    self.dataFileListDir + "/" + DATASET_LIST_INFERRING, 3, delimiter=",", prefix=self.datasetRootDir )
            else:
                raise Exception("Listing files on-the-fly not supported.")
            
            if ( 0 != self.dataEntries ):
                imgInferL = imgInferL[0:self.dataEntries]
                imgInferR = imgInferR[0:self.dataEntries]
                Q         = Q[0:self.dataEntries]

        # Dataloader.
        if ( self.flagGrayscale ):
            preprocessor = transforms.Compose( [ \
                PreProcess.NormalizeGray_OCV_naive(255, 0.5), \
                transforms.ToTensor(), \
                PreProcess.SingleChannel() ] )
        else:
            preprocessor = transforms.Compose( [ \
                PreProcess.NormalizeRGB_OCV(1.0/255), \
                transforms.ToTensor() ] )
        
        preprocessorGrad = transforms.Compose( [ \
            PreProcess.NormalizeGray_OCV_naive(1020, 0.0), \
            transforms.ToTensor(), \
            PreProcess.SingleChannel() ] )

        preprocessorDisp = transforms.Compose( [ \
            transforms.ToTensor() ] )

        if ( not self.flagInfer ):
            self.datasetTrain = DA.myImageFolder( imgTrainL, imgTrainR, dispTrain, True, \
                preprocessorImg=preprocessor, preprocessorGrad=preprocessorGrad, preprocessorDisp=preprocessorDisp, \
                cropSize=self.dlCropTrain, newSize=self.dlResize, gNoiseWidth=0 )
            self.datasetTest  = DA.myImageFolder( imgTestL,  imgTestR,  dispTest, False, \
                preprocessorImg=preprocessor, preprocessorGrad=preprocessorGrad, preprocessorDisp=preprocessorDisp, \
                cropSize=self.dlCropTest, newSize=self.dlResize, gNoiseWidth=0 )

            self.imgTrainLoader = torch.utils.data.DataLoader( \
                self.datasetTrain, \
                batch_size=self.dlBatchSize, shuffle=self.dlShuffle, num_workers=self.dlNumWorkers, drop_last=self.dlDropLast )

            self.imgTestLoader = torch.utils.data.DataLoader( \
                self.datasetTest, \
                batch_size=1, shuffle=False, num_workers=self.dlNumWorkers, drop_last=self.dlDropLast )
        else:
            self.datasetInfer = DA.inferImageFolder( imgInferL,  imgInferR, Q, \
                preprocessor=preprocessor, cropSize=self.dlCropTest, newSize=self.dlResize )

            self.imgInferLoader = torch.utils.data.DataLoader( \
                self.datasetInfer, \
                batch_size=1, shuffle=False, num_workers=self.dlNumWorkers, drop_last=self.dlDropLast )

        if ( self.flagGrayscale ):
            self.datasetTrain.enable_gray()
            self.datasetTest.enable_gray()

        if ( self.flagSobelX ):
            self.datasetTrain.enable_grad_x()
            self.datasetTrain.enable_grad_x()

    def init_model(self):
        raise Exception("init_model() virtual interface.")

    def post_init_model(self):
        if ( not self.flagCPU ):
            if ( True == self.multiGPUs ):
                self.model = nn.DataParallel(self.model)

            self.model.cuda()
    
    def set_optimizer_type(self, t):
        self.optType = t

    def init_optimizer(self):
        raise Exception("init_optimizer() virtual interface.")