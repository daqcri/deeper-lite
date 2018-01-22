local sys = require 'sys'
local csvigo = require 'csvigo'
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
--require 'dp'


local word2vec = nil
local t = sys.clock()

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-positivePairsTrainingFile','' , 'Positive Pairs Training File')
cmd:option('-negativePairsTrainingFile','' , 'Negative Pairs Training File')

cmd:option('-positivePairsDevFile','' , 'Positive Pairs Development File')
cmd:option('-negativePairsDevFile','' , 'Negative Pairs Development File')

cmd:option('-positivePairsTestingFile','' , 'Positive Pairs Testing File')
cmd:option('-negativePairsTestingFile','' , 'Negative Pairs Testing File')


cmd:option('-positivePairsTrainingFileBin','' , 'Positive Pairs Training Binary File')
cmd:option('-negativePairsTrainingFileBin','' , 'Negative Pairs Training Binary File')

cmd:option('-positivePairsDevFileBin','' , 'Positive Pairs Development Binary File')
cmd:option('-negativePairsDevFileBin','' , 'Negative Pairs Development Binary File')

cmd:option('-positivePairsTestingFileBin','' , 'Positive Pairs Testing Binary File')
cmd:option('-negativePairsTestingFileBin','' , 'Negative Pairs Testing Binary File')


cmd:option('-testMapFileBin','' , 'Index to Id Map for Test Data')
cmd:option('-debug_file_path','' , 'false positives and false negatives')
cmd:option('-test_predictions_file_path','' , 'path to test predictions file')
cmd:option('-perf_file_path','' , 'false positives and false negatives')
cmd:option('-threshold_file_path','' , 'save path to the negative drop threshold')



cmd:option('-firstDataFile','' , 'First Data File')
cmd:option('-secondDataFile','' , 'Second Data File')
cmd:option('-initEmbeddingsSaveFile','' , 'Vocab Embeddings Save File')


cmd:option('-numTableFields', 0, 'number of columns of the table')
cmd:option('-preTrainedModel', '', 'glove.840B.300d | GoogleNews-vectors-negative300')
cmd:option('-embeddingSize', 0, 'Word Embedding Size')

cmd:option('-seed', 0, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 0, 'number of threads')
cmd:option('-size', '', 'how many samples do we load: small | full | extra')
cmd:option('-model', '', 'type of model to construct: linear | mlp | convnet')
cmd:option('-loss', '', 'type of loss function to minimize: nll | mse | margin')
cmd:option('-save', '', 'subdirectory to save/log experiments in')
cmd:option('-plot', '', 'live plot yes | no')
cmd:option('-optimization', '', 'optimization method: Adam | Adagrad| ASGD | CG | LBFGS')
cmd:option('-hiddenX', 0, 'hidden to input size ratio')
cmd:option('-learningRate', 0, 'learning rate at t=0')
cmd:option('-learningRateDropThreshold', 0, 'perf metric plateuing threshold')
cmd:option('-learningRateDropCheckEpochs', 0, 'check for pleateau every ...')
cmd:option('-learningRateDropRatio', 0, 'ratio to drop lr by')
cmd:option('-batchSize', 0, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-maxIter', 0, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-minfreq', 0, 'minimum freq of a word in a corpus to be considered')
cmd:option('-percent_neg_train', 0, 'percentage of training data to be used')
cmd:option('-simMeasure', '', 'cosineDiff | diff | cosineDiff')
cmd:option('-computeFeatures', '', 'recompute features anew do not load from saved yes | no')
cmd:option('-noiseFlipLabels', '', 'noise labels yes | no')
cmd:option('-noiseFlipLabelsRatio', 0, '0.1 - 1')
cmd:option('-threshold',0, 'Negative Cosine Similarity Sampling Threshold [-1 , 1]')
cmd:option('-empty_cosine_penalty','0' , 'Empty Cosine Penalty')
cmd:option('-opMode', '', 'operation mode train_test | test')

cmd:text()
local opt = cmd:parse(arg or {})

print(opt)


torch.setdefaulttensortype('torch.FloatTensor')

if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

local function split(input, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    for str in string.gmatch(input, "([^"..sep.."]+)") do
        t[i] = str; i = i + 1
    end
    return t
end

local function print_stats(negativeTrainingTensor,positiveTrainingTensor, negativeDevTensor, positiveDevTensor, negativeTestingTensor,positiveTestingTensor)
  --========== Neg Train Metric ========================
  
  --Training Stats
  negTrainingTensorSize = negativeTrainingTensor:size()
  negTrainingMetric = torch.Tensor(negTrainingTensorSize[1])
  for i = 1,negTrainingTensorSize[1] do
    negTrainingMetric[i] = negativeTrainingTensor[{{i},{1,4}}]:mean()
  end

  print('\n\n\n ====== Training Stats ====== \n')
  print('\nCosine Training Negative Min')
  print(negTrainingMetric:min())
  print('\nCosine Training Negative Mean')
  print(negTrainingMetric:mean())
  print('\nCosine Training Negative Max')
  print(negTrainingMetric:max())
  print('\nCosine Training Negative Median')
  print(negTrainingMetric:median(1)[1])

  posTrainingTensorSize = positiveTrainingTensor:size()
  posTrainingMetric = torch.Tensor(posTrainingTensorSize[1])
  for i = 1,posTrainingTensorSize[1] do
    posTrainingMetric[i] = positiveTrainingTensor[{{i},{1,4}}]:mean()
  end
  print('\n\nCosine Training Positive Min')
  print(posTrainingMetric:min())
  print('\nCosine Training Positive Mean')
  print(posTrainingMetric:mean())
  print('\nCosine Training Positive Max')
  print(posTrainingMetric:max())
  print('\nCosine Training Positive Median')
  print(posTrainingMetric:median(1)[1])


  --Testing Stats
  negTestingTensorSize = negativeTestingTensor:size()
  negTestingMetric = torch.Tensor(negTestingTensorSize[1])
  for i = 1,negTestingTensorSize[1] do
    negTestingMetric[i] = negativeTestingTensor[{{i},{1,4}}]:mean()
  end

  print('\n\n\n ====== Testing Stats ====== \n')
  print('\nCosine Testing Negative Min')
  print(negTestingMetric:min())
  print('\nCosine Testing Negative Mean')
  print(negTestingMetric:mean())
  print('\nCosine Testing Negative Max')
  print(negTestingMetric:max())
  print('\nCosine Testing Negative Median')
  print(negTestingMetric:median(1)[1])

  posTestingTensorSize = positiveTestingTensor:size()
  posTestingMetric = torch.Tensor(posTestingTensorSize[1])
  for i = 1,posTestingTensorSize[1] do
    posTestingMetric[i] = positiveTestingTensor[{{i},{1,4}}]:mean()
  end
  print('\n\nCosine Testing Positive Min')
  print(posTestingMetric:min())
  print('\nCosine Testing Positive Mean')
  print(posTestingMetric:mean())
  print('\nCosine Testing Positive Max')
  print(posTestingMetric:max())
  print('\nCosine Testing Positive Median')
  print(posTestingMetric:median(1)[1])

  --Dev Stats
  negDevTensorSize = negativeDevTensor:size()
  negDevMetric = torch.Tensor(negDevTensorSize[1])
  for i = 1,negDevTensorSize[1] do
    negDevMetric[i] = negativeDevTensor[{{i},{1,4}}]:mean()
  end

  print('\n\n\n ====== Dev Stats ====== \n')
  print('\nCosine Dev Negative Min')
  print(negDevMetric:min())
  print('\nCosine Dev Negative Mean')
  print(negDevMetric:mean())
  print('\nCosine Dev Negative Max')
  print(negDevMetric:max())
  print('\nCosine Dev Negative Median')
  print(negDevMetric:median(1)[1])

  posDevTensorSize = positiveDevTensor:size()
  posDevMetric = torch.Tensor(posDevTensorSize[1])
  for i = 1,posDevTensorSize[1] do
    posDevMetric[i] = positiveDevTensor[{{i},{1,4}}]:mean()
  end
  print('\n\nCosine Dev Positive Min')
  print(posDevMetric:min())
  print('\nCosine Dev Positive Mean')
  print(posDevMetric:mean())
  print('\nCosine Dev Positive Max')
  print(posDevMetric:max())
  print('\nCosine Dev Positive Median')
  print(posDevMetric:median(1)[1])
 
  
  local cut_off_cosine = math.min(posTestingMetric:min(), posTrainingMetric:min(), posDevMetric:min()) 
  return cut_off_cosine

end

local function FillDataTable(pairsTable, firstTable, secondTable, indextoIdTable, indextoIdTableCounter)
  
  local pairsDataTable = {}
  for i=1, #pairsTable do
    xlua.progress(i,#pairsTable)
    local pairFirstData = firstTable[pairsTable[i][1]]
    local pairSecondData = secondTable[pairsTable[i][2]]
    pairsDataTable[i] = {pairFirstData, pairSecondData}
    if indextoIdTable ~= nil then
      indextoIdTable[indextoIdTableCounter] = {pairsTable[i][1], pairsTable[i][2]}
      indextoIdTableCounter = indextoIdTableCounter + 1
    end
  end
  return pairsDataTable
end


local cosine = nn.CosineDistance()

local function ExtractFeatures(dataTable, negativeSamplingThreshold, emtpyCosinePenalty)
  local firstFeaturesTensor = torch.FloatTensor(opt.numTableFields,opt.embeddingSize)
  local secondFeaturesTensor = torch.FloatTensor(opt.numTableFields,opt.embeddingSize)
  local dataTensor = nil
  if opt.simMeasure == 'cosine' or opt.simMeasure == 'diff' then
    --print('Computing ' .. opt.simMeasure .. ' similairty measure ...\n')
    dataTensor = torch.FloatTensor(#dataTable, opt.numTableFields)
  elseif opt.simMeasure == 'cosineDiff' then
    --print('Computing cosine and diff similarity measures \n')
    dataTensor = torch.FloatTensor(#dataTable, 2*opt.numTableFields)
  else
    error('Undefined Similarity Measure')
  end
  local included = torch.Tensor(dataTensor:size()[1])
  local numIncluded = 0
  for i = 1, #dataTable do
    xlua.progress(i,#dataTable)
    for j=1, opt.numTableFields do
      firstFeaturesTensor:zero()
      secondFeaturesTensor:zero()
      local sentence1 = dataTable[i][1][j]
      local numWordsInSentences = 0
      for word in sentence1:gmatch("%w+") do --TODO: very naive to break on white space only
        firstFeaturesTensor[j]:add(word2vec:word2vec(word))
        numWordsInSentences = numWordsInSentences + 1
      end
      firstFeaturesTensor[j]:div(numWordsInSentences)

      local sentence2 = dataTable[i][2][j]
      
      
      numWordsInSentences = 0
      for word in sentence2:gmatch("%w+") do --TODO: very naive to break on white space only, use porter or something more advanced
        secondFeaturesTensor[j]:add(word2vec:word2vec(word))
        numWordsInSentences = numWordsInSentences + 1
      end
      secondFeaturesTensor[j]:div(numWordsInSentences)
      local dist = cosine:forward{firstFeaturesTensor[j],secondFeaturesTensor[j]}

      --local diff = torch.norm((firstFeaturesTensor[j] - secondFeaturesTensor[j]),2)
      local diff = nn.Abs()(firstFeaturesTensor[j] - secondFeaturesTensor[j])
      diff = torch.norm(diff)
      if dist[1]~=dist[1] then  --cosine is NaN
        if string.len(sentence1) == 0 and string.len(sentence2) == 0 then
          dist = 1
        elseif (string.len(sentence1) == 0 and string.len(sentence2) < 3) or (string.len(sentence1) < 3 and string.len(sentence2) == 0) then
          dist = 1
        else
          dist = emtpyCosinePenalty  -- cosine is NaN, so assigning worst value from a cosine similarity point of view
        end
      end  

      if diff~=diff then  --diff is NaN, this is mostly due to a mising attribute(sentence)
        if string.len(sentence1) == 0 and string.len(sentence2) == 0 then
          diff = 0
        elseif sentence1~="" and sentence2 =="" then 
          diff = torch.norm(firstFeaturesTensor[j])
          diff = diff / string.len(sentence1)
        elseif sentence1=="" and sentence2 ~="" then
          diff = torch.norm(secondFeaturesTensor[j])
          diff = diff / string.len(sentence2)
        end
      end

     if opt.simMeasure == 'cosine' then
      dataTensor[i][j] = dist
     elseif opt.simMeasure == 'diff' then
      dataTensor[i][j] = diff
     elseif opt.simMeasure == 'cosineDiff' then
      dataTensor[i][j] = dist
      dataTensor[i][j+opt.numTableFields] = diff  
     end
   end
   
   if dataTensor[ {{i},{1,4} } ]:mean() < negativeSamplingThreshold then
    included[i] = 0
   else
    included[i] = 1
    numIncluded = numIncluded + 1
   end
  end
  return dataTensor, included, numIncluded
end

local positiveTrainingTensor = nil
local negativeTrainingTensor = nil
local positiveDevTensor = nil
local negativeDevTensor = nil
local positiveTestingTensor = nil
local negativeTestingTensor = nil
local testIndexToIdTable = {}
local debug_file = torch.DiskFile(opt.debug_file_path, 'w')
local test_predictions_file = torch.DiskFile(opt.test_predictions_file_path, 'w')

if opt.computeFeatures == 'yes' then --   not paths.filep(opt.positivePairsTrainingFileBin) then
  sys.tic()
  if opt.preTrainedModel == 'glove.840B.300d' then
    print("Loading Pre-Trained GloVe word2vec Words Embeddings ... \n")
    word2vec = dofile('glove/glove.lua')
    t = sys.toc()    
    print("Loaded Pre-Trained GloVe word2vec in: " .. t .. " seconds\n")
  elseif opt.preTrainedModel == 'GoogleNews-vectors-negative300' then
    print("Loading Pre-Trained Google News word2vec Words Embeddings ... \n")
    word2vec = dofile('GoogleNews/w2vutils.lua')
    t = sys.toc()    
    print("Loaded Pre-Trained Googele News word2vec took: " .. t .. " seconds\n")
  end
  

  sys.tic()
  
  local positivePairsTraining = nil
  local negativePairsTraining = nil
  local positivePairsDev = nil
  local negativePairsDev = nil

  if opt.opMode == 'train_test' then
    print("Loading Raw Train Data .. \n")
    positivePairsTraining = csvigo.load({path = opt.positivePairsTrainingFile, mode = "large"})
    negativePairsTraining = csvigo.load({path = opt.negativePairsTrainingFile, mode = "large"})

    print("Loading Raw Dev Data .. \n")
    positivePairsDev = csvigo.load({path = opt.positivePairsDevFile, mode = "large"})
    negativePairsDev = csvigo.load({path = opt.negativePairsDevFile, mode = "large"})
  end
  local positivePairsTesting = csvigo.load({path = opt.positivePairsTestingFile, mode = "large"})
  local negativePairsTesting = csvigo.load({path = opt.negativePairsTestingFile, mode = "large"})
  print("Loading Raw Test Data .. \n")
  local firstData = csvigo.load({path = opt.firstDataFile, mode = "large"})
  local secondData = csvigo.load({path = opt.secondDataFile, mode = "large"})

  t = sys.toc()    
  print("Data Loading took: " .. t .. " seconds\n")

  print("Preprocessing data .. \n")
  sys.tic()
  local firstDataTable = {}
  local secondDataTable = {}

  for i=1,#firstData do
    local key
    local value = {}
    key = firstData[i][1]
    for j=2,#firstData[i] do
      table.insert(value,firstData[i][j])
    end
    firstDataTable[key] = value
  end

  for i=1,#secondData do
    local key
    local value = {}
    key = secondData[i][1]
    for j=2,#secondData[i] do
      table.insert(value,secondData[i][j])
    end
    secondDataTable[key] = value
  end
  t = sys.toc()    
  print("Data Preprocessing took: " .. t .. " seconds\n")
  
  print("Computing Features ... ")
  sys.tic()
  
  local positivePairsTrainingTable = nil
  local negativePairsTrainingTable = nil
  local positivePairsDevTable = nil
  local negativePairsDevTable = nil
  local includedNegativeTraining = 0
  local includedNegativeDev = 0
  local includedNegativeTesting = 0

  if opt.opMode == 'train_test' then
    print('Filling Train Data Tables ...')
    positivePairsTrainingTable = FillDataTable(positivePairsTraining, firstDataTable, secondDataTable)
    negativePairsTrainingTable = FillDataTable(negativePairsTraining, firstDataTable, secondDataTable)
    print('Extracting Features from Train Data Tables ...')
    positiveTrainingTensor = ExtractFeatures(positivePairsTrainingTable,-1,opt.empty_cosine_penalty) -- -1 as we never drop positives, 0 as we lightly penalize empty positives
    negativeTrainingTensorFull, negativeTrainingIncluded, numNegativeTrainingIncluded = ExtractFeatures(negativePairsTrainingTable, opt.threshold,-1)
    print('Filtering Train Tensors by Cosine Similarity Threshold of: ' .. opt.threshold)
    negativeTrainingTensor = torch.Tensor(numNegativeTrainingIncluded, negativeTrainingTensorFull:size()[2])
    local negativeCounter = 0
    for i = 1, negativeTrainingTensorFull:size()[1] do
      if negativeTrainingIncluded[i] == 1 then
        negativeCounter = negativeCounter + 1
        negativeTrainingTensor[negativeCounter] = negativeTrainingTensorFull[i]
      end
    end
    print("negativeTrainingTensorFull:size() before cosine cut: " .. negativeTrainingTensorFull:size()[1])
    print("negativeTrainingTensor:size() after cosine cut: " .. negativeTrainingTensor:size()[1])
    print("positiveTrainingTensor:size(): " .. positiveTrainingTensor:size()[1])
    print('\n')

    print('Filling Dev Data Tables ...')
    positivePairsDevTable = FillDataTable(positivePairsDev, firstDataTable, secondDataTable)
    negativePairsDevTable = FillDataTable(negativePairsDev, firstDataTable, secondDataTable)
    print('Extracting Features from Dev Data Tables ...')
    positiveDevTensor = ExtractFeatures(positivePairsDevTable,-1,opt.empty_cosine_penalty)
    negativeDevTensorFull, negativeDevIncluded, numNegativeDevIncluded = ExtractFeatures(negativePairsDevTable, opt.threshold,-1)
    print('Filtering Dev Tensors by Cosine Similarity Threshold of: ' .. opt.threshold)
    negativeDevTensor = torch.Tensor(numNegativeDevIncluded, negativeDevTensorFull:size()[2])
    negativeCounter = 0
    for i = 1, negativeDevTensorFull:size()[1] do
      if negativeDevIncluded[i] == 1 then
        negativeCounter = negativeCounter + 1
        negativeDevTensor[negativeCounter] = negativeDevTensorFull[i]
      end
    end
    print('\n')
    print("negativeDevTensorFull:size() before cosine cut: " .. negativeDevTensorFull:size()[1])
    print("negativeDevTensor:size() after cosine cut: " .. negativeDevTensor:size()[1])
    print("positiveDevTensor:size(): " .. positiveDevTensor:size()[1])
    print('\n')


    --balance negatives in training
    negativeTrainingTensor = negativeTrainingTensor[ { {1,math.floor(negativeTrainingTensor:size(1)*opt.percent_neg_train)}, {} } ]

    print("XXXXXXXXXX===================XXXXXXXXXXXXXXXXXXXXXX")
    print("Training Positive: " .. positiveTrainingTensor:size(1))
    print("Training Negative: " .. negativeTrainingTensor:size(1))
    print("XXXXXXXXXX===================XXXXXXXXXXXXXXXXXXXXXX")
  end
  
  print('Filling Test Data Tables ...')
  local positivePairsTestingTable = FillDataTable(positivePairsTesting, firstDataTable, secondDataTable,testIndexToIdTable, 1)
  local negativePairsTestingTable = FillDataTable(negativePairsTesting, firstDataTable, secondDataTable,testIndexToIdTable, 1+#positivePairsTesting)
  print('Extracting Features from Test Data Tables ...')
  positiveTestingTensor = ExtractFeatures(positivePairsTestingTable,-1,opt.empty_cosine_penalty)
  negativeTestingTensorFull, negativeTestingIncluded, numNegativeTestingIncluded = ExtractFeatures(negativePairsTestingTable, opt.threshold,-1)
  print('Filtering Test Tensors by Cosine Similarity Threshold of: ' .. opt.threshold)
  negativeTestingTensor = torch.Tensor(numNegativeTestingIncluded, negativeTestingTensorFull:size()[2])
  negativeCounter = 0
  for i = 1, negativeTestingTensorFull:size()[1] do
    if negativeTestingIncluded[i] == 1 then
      negativeCounter = negativeCounter + 1
      negativeTestingTensor[negativeCounter] = negativeTestingTensorFull[i]
      testIndexToIdTable[negativeCounter] = testIndexToIdTable[i]
    end
  end
  print("negativeTestingTensorFull:size() before cosine filtering: " .. negativeTestingTensorFull:size()[1])
  print("negativeTestingTensor:size() after cosine filtering: " .. negativeTestingTensor:size()[1])
  print("positiveTestingTensor:size(): " .. positiveTestingTensor:size()[1])
  print('\n')

  if opt.opMode == 'train_test' then
    local cut_off_cosine = print_stats(negativeTrainingTensorFull, positiveTrainingTensor, negativeDevTensorFull, positiveDevTensor,negativeTestingTensorFull, positiveTestingTensor)
    print("Safe Suggested Cut-Off Cosine Sim Value: " .. 0.9*cut_off_cosine)
    print("Current Cut-off cosine sim value: " .. opt.threshold)
    
    negativeTestSamplingReductionRatio = negativeTestingTensorFull:size()[1] / negativeTestingTensor:size()[1]
    negativeTestFilteredToPositveRatio = negativeTestingTensor:size()[1] / positiveTestingTensor:size()[1]

    print('Negative Test Sampling Reduction Ratio: ' .. negativeTestSamplingReductionRatio .. '\n')
    print('Negative Test to Positive After Filgering: ' .. negativeTestFilteredToPositveRatio .. '\n')
  end

  t= sys.toc()
 
  print("Computing Features took: " .. t)

  sys.tic()
  if opt.opMode == 'train_test' then
    torch.save(opt.positivePairsTrainingFileBin, positiveTrainingTensor)
    torch.save(opt.negativePairsTrainingFileBin, negativeTrainingTensor)

    torch.save(opt.positivePairsDevFileBin, positiveDevTensor)
    torch.save(opt.negativePairsDevFileBin, negativeDevTensor)
  end
  torch.save(opt.positivePairsTestingFileBin, positiveTestingTensor)
  torch.save(opt.negativePairsTestingFileBin, negativeTestingTensor)
  torch.save(opt.testMapFileBin, testIndexToIdTable)

  sys.toc()
  print("Saving Features took: " .. t)
else
  print("Loading Pre-Computed Google News word2vec Features ...")
  sys.tic()
  if opt.opMode == 'train_test' then
    positiveTrainingTensor = torch.load(opt.positivePairsTrainingFileBin)
    negativeTrainingTensor = torch.load(opt.negativePairsTrainingFileBin)

    positiveDevTensor = torch.load(opt.positivePairsDevFileBin)
    negativeDevTensor = torch.load(opt.negativePairsDevFileBin)
  end
  positiveTestingTensor = torch.load(opt.positivePairsTestingFileBin)
  negativeTestingTensor = torch.load(opt.negativePairsTestingFileBin)
  testIndexToIdTable = torch.load(opt.testMapFileBin)

  t = sys.toc()    
  print("Loading Pre-Computed Features took: " .. t .. " seconds\n")
end



local trainData = {}
local devData = {}

if opt.opMode == 'train_test' then
  trainData = {
     data = torch.cat(positiveTrainingTensor, negativeTrainingTensor,1),
     labels = torch.cat(torch.Tensor(positiveTrainingTensor:size(1)):fill(1), torch.Tensor(negativeTrainingTensor:size(1)):fill(2)),
     size = function() return positiveTrainingTensor:size(1) + negativeTrainingTensor:size(1) end
  }

  devData = {
     data = torch.cat(positiveDevTensor,negativeDevTensor,1),
     labels = torch.cat(torch.Tensor(positiveDevTensor:size(1)):fill(1),torch.Tensor(negativeDevTensor:size(1)):fill(2)),
     size = function() return positiveDevTensor:size(1) + negativeDevTensor:size(1) end
  }
end

local testData = {
   data = torch.cat(positiveTestingTensor,negativeTestingTensor,1),
   labels = torch.cat(torch.Tensor(positiveTestingTensor:size(1)):fill(1),torch.Tensor(negativeTestingTensor:size(1)):fill(2)),
   size = function() return positiveTestingTensor:size(1) + negativeTestingTensor:size(1) end
}

if opt.noiseFlipLabels == 'yes' then
  local numToFlipPos = math.floor(opt.noiseFlipLabelsRatio * positiveTrainingTensor:size(1))
  local numToFlipNeg = math.floor(opt.noiseFlipLabelsRatio * negativeTrainingTensor:size(1))
  local numFlippedPos = 0
  local numFlippedNeg = 0
  for i = 1, trainData.size() do
    local toFlip = torch.rand(1)
    if toFlip[1] > 0.5 then
      if trainData.labels[i] == 1 and numFlippedPos < numToFlipPos then
        trainData.labels[i] = 2
        numFlippedPos = numFlippedPos + 1
      elseif trainData.labels[i] == 2 and numFlippedNeg < numToFlipNeg then
        trainData.labels[i] = 1
        numFlippedNeg = numFlippedNeg + 1
      end
    end
  end
end


local noutputs = 2
local nfeats = opt.numTableFields

if opt.simMeasure == 'cosineDiff' then
  nfeats = 2*opt.numTableFields
end

local ninputs = nfeats
local nhiddens = {opt.hiddenX*opt.numTableFields, opt.hiddenX*opt.numTableFields, opt.hiddenX*opt.numTableFields}

if opt.model == 'linear' then
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

   best_dev_model = nn.Sequential()
   best_dev_model:add(nn.Reshape(ninputs))
   best_dev_model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens[1]))
   model:add(nn.ReLU())
   model:add(nn.Linear(nhiddens[1],nhiddens[2]))
   model:add(nn.ReLU())
   model:add(nn.Linear(nhiddens[2],nhiddens[3]))
   model:add(nn.ReLU())
   model:add(nn.Linear(nhiddens[3],noutputs))

   best_dev_model = nn.Sequential()
   best_dev_model:add(nn.Reshape(ninputs))
   best_dev_model:add(nn.Linear(ninputs,nhiddens[1]))
   best_dev_model:add(nn.ReLU())
   best_dev_model:add(nn.Linear(nhiddens[1],nhiddens[2]))
   best_dev_model:add(nn.ReLU())
   best_dev_model:add(nn.Linear(nhiddens[2],nhiddens[3]))
   best_dev_model:add(nn.ReLU())
   best_dev_model:add(nn.Linear(nhiddens[3],noutputs))

 else
   error('unsupported model')
end

if opt.loss == 'margin' then
   criterion = nn.MultiMarginCriterion()
elseif opt.loss == 'nll' then
   model:add(nn.LogSoftMax())
   --weighted loss for class imbalance, uncomment the following 4 lines of code
   -- weights = torch.Tensor(2)
   -- weights[1]=0.99
   -- weights[2]=0.01
   --criterion = nn.ClassNLLCriterion(weights,true,-100)
   criterion = nn.ClassNLLCriterion()
elseif opt.loss == 'mse' then
   model:add(nn.Tanh())
   criterion = nn.MSECriterion()
   criterion.sizeAverage = false
else
   error('unsupported')
end

if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

print(model)

classes = {'1','2'}

confusion = optim.ConfusionMatrix(classes)

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if model then
   parameters,gradParameters = model:getParameters()
end

if best_dev_model then
   bd_parameters,bd_gradParameters = best_dev_model:getParameters()
end
if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd
elseif opt.optimization == 'Adagrad' then
   optimState = {
      learningRate = opt.learningRate,
      --weightDecay = opt.weightDecay,
      --momentum = opt.momentum,
      --learningRateDecay = 1e-7
   }
   optimMethod = optim.adagrad
elseif opt.optimization == 'Adam' then
   optimState = {
      learningRate = opt.learningRate,
      --weightDecay = opt.weightDecay,
      --momentum = opt.momentum,
      --learningRateDecay = 1e-7
   }
   optimMethod = optim.adam

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trainData:size() * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unsupported optimization method')
end


local function train()
   epoch = epoch or 1
   local time = sys.clock()
   model:training()
   shuffle = torch.randperm(trainData:size())
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(), opt.batchSize do
      xlua.progress(t, trainData:size())
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         if opt.type == 'double' then
            input = input:double()
            if opt.loss == 'mse' then
               target = target:double()
            end
         elseif opt.type == 'cuda' then
            input = input:cuda();
            if opt.loss == 'mse' then
               target = target:cuda()
            end
         end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      local feval = function(x)
                       if x ~= parameters then
                          parameters:copy(x)
                       end
                       gradParameters:zero()
                       local f = 0
                       for i = 1,#inputs do
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)
                          confusion:add(output, targets[i])
                       end
                       gradParameters:div(#inputs)
                       f = f/#inputs
                       return f,gradParameters
                    end
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print("TRAINING CONFUSION START")
   print(confusion)
   print("TRAINING CONFUSION END")

   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot == 'yes' then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   confusion:zero()
end

local plateau = torch.Tensor(opt.learningRateDropCheckEpochs)
local f1Tensor = torch.DoubleTensor(opt.maxIter):zero()
local best_dev_model_file_path = paths.concat(opt.save, 'best_dev_model.net')
local test_confusion = torch.Tensor(#classes, #classes)
local maxEpoch = 0

local falsePositives = {}
local falseNegatives = {}
local test_predictions = {}

local function evaluate(data, model, is_dev)
   local time = sys.clock()
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   model:evaluate()
   print('==> evaluating model:')
   for t = 1,data:size() do
      xlua.progress(t, data:size())
      local input = data.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = data.labels[t]
      local pred = model:forward(input)
      if is_dev == false then
        if pred[1] > pred[2] then
          label = 1
        else
          label = 2
        end
        table.insert(test_predictions, {testIndexToIdTable[t], label})  
        if pred[1] > pred[2] and target == 2 then
          table.insert(falsePositives,testIndexToIdTable[t])
        elseif pred[1] < pred[2] and target == 1 then
          table.insert(falseNegatives, testIndexToIdTable[t])
        end
      end

      confusion:add(pred, target)
   end
   time = sys.clock() - time
   time = time / data:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   if is_dev == true then 
     print(confusion)
   end

   local precision = confusion.mat[{1,1}]*100/(confusion.mat[{1,1}] + confusion.mat[{2,1}])
   local recall = confusion.mat[{1,1}]*100/(confusion.mat[{1,1}] + confusion.mat[{1,2}])
   local f1 = 2*precision*recall/(precision+recall)


   if is_dev == true then
    print('Dev P = ' .. precision)
    print('Dev R = ' .. recall)
    print('Dev F1 = ' .. f1)
   end
   -- testLogger:add{['P,R,F1'] = precision .. ', ' .. recall .. ', ' .. f1}
   if is_dev == false then
      local copy_conf = torch.Tensor(#classes, #classes)
      copy_conf:copy(confusion.mat:float())
     return precision, recall, f1, copy_conf
   end
   if(f1 == f1) then
    local max_F1 = f1Tensor:max()
    if f1 > max_F1 then
      bd_parameters:copy(parameters)
      maxEpoch = epoch
    end
    f1Tensor[epoch] = f1
   end
   --print(f1Tensor:max())
   testLogger:add{['% mean class accuracy (dev set)'] = confusion.totalValid * 100}  
   if opt.plot == 'yes' then
      testLogger:style{['% mean class accuracy (dev set)'] = '-'}
      testLogger:plot()
   end
   
   -- if epoch % opt.learningRateDropCheckEpochs == 0 and epoch > opt.learningRateDropCheckEpochs then
   --  optimState.weightDecay = 0
   -- end
   plateau[epoch%opt.learningRateDropCheckEpochs+1] = confusion.totalValid
   if epoch % opt.learningRateDropCheckEpochs == 0 then
      local varPlateau = torch.var(plateau)
      print('epoch: ' .. varPlateau .. ' , ' .. opt.learningRateDropThreshold)
      if varPlateau < opt.learningRateDropThreshold then
        --print(varPlateau) 
        print(optimState)
        optimState.learningRate = optimState.learningRate*opt.learningRateDropRatio
        print(optimState)
        plateau:zero()
      end
    end
   if average then
      parameters:copy(cachedparams)
   end
   confusion:zero()
   epoch = epoch + 1
end

if opt.opMode == 'train_test' then
  for i=1,opt.maxIter do
     train()
     evaluate(devData,model,true)
  end
  torch.save(best_dev_model_file_path, best_dev_model)
end


confusion:zero()
if opt.opMode == 'test' then
  best_dev_model = torch.load(best_dev_model_file_path)
end
print('\n\nUsing best dev model on test set:\n\n')
local test_p, test_r, test_f1, test_confusion = evaluate(testData, best_dev_model, false)
 
-- negativeSamplingReductionRatio = negativeTestingTensorFull:size()[1] / negativeTestingTensor:size()[1]
-- negativeToPositveRatio = negativeTestingTensor:size()[1] / positiveTestingTensor:size()[1]

-- print('Negative Sampling Reduction Ratio: ' .. negativeSamplingReductionRatio .. '\n')
-- print('Negative to Positive After Reduction: ' .. negativeToPositveRatio .. '\n')



print('\n\nTest Resutls:\n\n')
print('Test P Score: ' .. test_p .. '\n')
print('Test R Score: ' .. test_r .. '\n')
print('Test F1 Score: ' .. test_f1 .. '\n')
print('Test Confusion: \n')
print(test_confusion)
local perf_file = torch.DiskFile(opt.perf_file_path, 'w')
perf_file:writeString('Test F1 Score: ' .. test_f1 .. '\n')
perf_file:writeString('Test Confusion: \n')
perf_file:writeString(test_confusion[{1,1}] .. '    ' .. test_confusion[{1,2}] .. '\n')
perf_file:writeString(test_confusion[{2,1}] .. '    ' .. test_confusion[{2,2}] .. '\n')
perf_file:close()


local threshold_file = torch.DiskFile(opt.threshold_file_path, 'w')
print(opt.threshold)
threshold_file:writeFloat(opt.threshold)
threshold_file:close()


for k,v in ipairs(test_predictions) do
  test_predictions_file:writeString('"' .. v[1][1] .. '"' .. ',' .. '"' .. v[1][2] .. '"' .. ',' .. v[2] .. '\n')
end
test_predictions_file:close()


debug_file:writeString('Positive Test Cases: ' .. positiveTestingTensor:size(1) .. '\n')
debug_file:writeString('Negative Test Cases: ' .. negativeTestingTensor:size(1) .. '\n')
debug_file:writeString('\nFalse Positives: ' .. #falsePositives .. '\n')
for k,v in ipairs(falsePositives) do
  debug_file:writeString('"' .. v[1] .. '"' .. ',' .. '"' .. v[2] .. '"' .. '\n')
end
debug_file:writeString('\n')
debug_file:writeString('\n\nFalse Negatives: ' .. #falseNegatives .. '\n')
for k,v in ipairs(falseNegatives) do
  debug_file:writeString('"' .. v[1] .. '"' .. ',' .. '"' .. v[2] .. '"' .. '\n')
end
debug_file:close()
