local glove = dofile('glove.lua')
local k = 3
hellorep = glove:word2vec('2007')
helloneighbors = glove:distance(hellorep,k)
print(torch.mean(hellorep))
print(torch.sum(hellorep))
print(helloneighbors)

hirep = glove:word2vec('14')
hineighbors = glove:distance(hirep,k)
print(torch.mean(hirep))
print(torch.sum(hirep))
print(hineighbors)

