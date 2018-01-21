local sys = require 'sys'
local csvigo = require 'csvigo'
local threads = require 'threads'

require 'torch'
require 'nn'
require 'xlua'
require 'optim'
--require 'dp'

local total_time = sys.clock()

local word2vec = nil
local t = sys.clock()

cmd = torch.CmdLine()
cmd:text('Options:')

cmd:option('-predPairsFile','' , 'Prediction Pairs  File')
cmd:option('-predPairsFileBin','' , 'Prediction Pairs  Binary File')
cmd:option('-predMapFileBin','' , 'Index to Id Map for Prediction Data')
cmd:option('-predictions_file_path','' , 'path to predictions file')
cmd:option('-firstDataFile','' , 'First Data File')
cmd:option('-secondDataFile','' , 'Second Data File')
cmd:option('-numTableFields', 0, 'number of columns of the table')
cmd:option('-preTrainedModel', '', 'glove.840B.300d | GoogleNews-vectors-negative300')
cmd:option('-embeddingSize', 0, 'Word Embedding Size')
cmd:option('-seed', 0, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-save', '', 'subdirectory to save/log experiments in')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-simMeasure', '', 'cosineDiff | diff | cosineDiff')
cmd:option('-threshold_file_path','' , 'save path to the negative drop threshold')
cmd:option('-computeFeatures', '', 'recompute features anew do not load from saved yes | no')

cmd:text()
local opt = cmd:parse(arg or {})

local threshold_file = torch.DiskFile(opt.threshold_file_path, 'r')
opt.threshold = threshold_file:readFloat()
threshold_file:close()

print(opt)

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




local nthread = 8
local njob = nthread

local function TableLength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
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
    xlua.progress(i, #dataTable)
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

local predTensor = nil
local predIndexToTable = {}
local test_predictions_file = torch.DiskFile(opt.predictions_file_path, 'w')

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
  

  print("Loading data .. \n")
  sys.tic()
  local allPairs = csvigo.load({path = opt.predPairsFile, mode = "large"})

  local firstData = csvigo.load({path = opt.firstDataFile, mode = "large"})
  local secondData = csvigo.load({path = opt.secondDataFile, mode = "large"})

  t = sys.toc()    
  print("Data Loading took: " .. t .. " seconds\n")

  print("Preprocessing data .. \n")
  sys.tic()
  local firstDataTable = {}
  local secondDataTable = {}

  for i=2,#firstData do
    local key
    local value = {}
    key = firstData[i][1]
    for j=2,#firstData[i] do
      table.insert(value,firstData[i][j])
    end
    firstDataTable[key] = value  
  end

  for i=2,#secondData do
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
  
  print("Filling Data Table  ... ")
  sys.tic()
  

  print("Memory before FillDataTable: " .. collectgarbage("count")/1000000 .. " GB")
  local predPairsTable = FillDataTable(allPairs, firstDataTable, secondDataTable, predIndexToTable,1)
  firstDataTable = nil
  secondDataTable =nil
  print("Memory after FillDataTable before collect: " .. collectgarbage("count")/1000000 .. " GB")
  collectgarbage()
  print("Memory after FillDataTable after collect: " .. collectgarbage("count")/1000000 .. " GB")
  t = sys.toc()
  print("Filling Data Table took: " .. t .. 'seconds\n')

  print("Extracting Features from Data Table ...")
  sys.tic()

  predTensorFull,included, numIncluded = ExtractFeatures(predPairsTable, opt.threshold,-1)
  
  t = sys.toc()
  print("Extracting Features from Data Table took: " .. t .."seconds\n")
  
  word2vec = nil
  predPairsTable = nil
  print("Memory after ExtractFeatures before collect: " .. collectgarbage("count")/1000000 .. " GB")
  collectgarbage()
  print("Memory after ExtractFeatures after collect: " .. collectgarbage("count")/1000000 .. " GB")


  print('size before cosine sim thresholding: ' .. predTensorFull:size()[1])

  sys.tic()
  print("Filtering by cosine threshold: " .. opt.threshold)
  predTensor = torch.Tensor(numIncluded, predTensorFull:size()[2])
  --reducedPredIndexToTable = {}
  local predCounter = 0
  for i = 1, predTensorFull:size()[1] do
    if included[i] == 1 then
      predCounter = predCounter + 1
      predTensor[predCounter] = predTensorFull[i]
      predIndexToTable[predCounter] = predIndexToTable[i]
    end
  end
  --predIndexToTable = reducedPredIndexToTable

  print('size after cosine sim thresholding: ' .. numIncluded)  
  print('size after cosine sim thresholding: (check): ' .. predTensor:size()[1])
  
  print('\n')
  t = sys.toc()
  print("Filtering by cosine threshold took: " .. t .. ' seconds\n')
  
  sys.tic()
  torch.save(opt.predPairsFileBin, predTensor)
  torch.save(opt.predMapFileBin, predIndexToTable)
  sys.toc()
  print("Saving Features took: " .. t)
else
  print("Loading Pre-Computed Google News word2vec Features ...")
  sys.tic()
  predTensor = torch.load(opt.predPairsFileBin)
  predIndexToTable = torch.load(opt.predMapFileBin)
  t = sys.toc()    
  print("Loading Pre-Computed Features took: " .. t .. " seconds\n")
end

local predData = {
   data = predTensor:float(),
   size = function() return  predTensor:size(1) end
}

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
local test_predictions = {}

local function evaluate(data, model, is_dev)
   local time = sys.clock()
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   model:evaluate()
   print('==> Finding matches:\n')
   for t = 1,data:size() do
      xlua.progress(t, data:size())
      local input = data.data[t]
      
      -- if opt.type == 'double' then input = input:double()
      -- elseif opt.type == 'float' then input = input:float()
      -- elseif opt.type == 'cuda' then input = input:cuda() end
      
      local pred = model:forward(input)
      if pred[1] > pred[2] then
        label = 1
      else
        label = 2
      end
      table.insert(test_predictions, {predIndexToTable[t], label})  
   end
   time = sys.clock() - time
   time = time / data:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   
end

local best_dev_model_file_path = paths.concat(opt.save, 'best_dev_model.net') 
best_dev_model = torch.load(best_dev_model_file_path)

if opt.type == 'cuda' then
   best_dev_model:cuda()
end

local test_p, test_r, test_f1, test_confusion = evaluate(predData, best_dev_model, false)

local count_matches = 0
for k,v in ipairs(test_predictions) do
  if v[2] == 1 then
    count_matches = count_matches + 1
    test_predictions_file:writeString(v[1][1]  .. ','  .. v[1][2] ..'\n')
  end
end

test_predictions_file:close()
total_time = sys.clock() - total_time
print("\n==> Total time = " .. (total_time) .. ' seconds, predicted: ' .. (predData:size()) .. ' pairs\n' )
print("(" .. count_matches .. ")".. " MATCHES FOUND\n")
