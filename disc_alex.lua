require 'cudnn'
require 'cunn'
require 'nn'

local spatialConv = cudnn.SpatialConvolution
local spatialMaxPool = cudnn.SpatialMaxPooling
local features  = nn.Sequential()
local classifier = nn.Sequential()
local batchNorm = nn.SpatialBatchNormalization

features:add(spatialConv(3,64,11,11,4,4,2,2))	--224 -> 55
features:add(spatialMaxPool(3,3,2,2))
features:add(cudnn.ReLU(true))
features:add(batchNorm(64,nil,nil,false))

features:add(spatialConv(64,192,5,5,1,1,2,2))	--27 -> 27
features:add(spatialMaxPool(3,3,2,2))           --27 -> 13
features:add(cudnn.ReLU(true))
features:add(batchNorm(192,nil,nil,false))

features:add(spatialConv(192,384,3,3,1,1,1,1))  --13 -> 13
features:add(cudnn.ReLU(true))
features:add(batchNorm(384,nil,nil,false))

features:add(spatialConv(384,256,3,3,1,1,1,1))  --13 -> 13
features:add(cudnn.ReLU(true))
features:add(batchNorm(256,nil,nil,false))

features:add(spatialConv(256,256,3,3,1,1,1,1))  --13 -> 13
features:add(spatialMaxPool(3,3,2,2))           --13 -> 6
features:add(cudnn.ReLU(true))
features:add(batchNorm(256,nil,nil,false))


classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(batchNorm(4096,nil,nil,false))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(batchNorm(4096,nil,nil,false))
classifier:add(nn.Linear(4096,1000))
classifier:add(nn.LogSoftMax())

local disc_model = nn.Sequential()

disc_model:add(features):add(classifier)






