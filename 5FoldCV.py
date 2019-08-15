# -*- coding: utf-8 -*-
import codecs
import random
import re
import sys
import os
import time
from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Process, Queue
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import numpy as np


class AsymModel(object):

    def __init__(self, array, directory, numFeature, numFold):
        """
        Construction function. Init variables:
        """
        self.array = array
        self.numRow, self.numCol = array.shape
        self.directory = directory
        self.numFold = numFold
        self.numFeature = numFeature

        disease = np.loadtxt(directory + "Gaussian_disease.csv", dtype=np.float, delimiter=",")
        miRNA = np.loadtxt(directory + "Gaussian_miRNA.csv", dtype=np.float, delimiter=",")
        top = np.hstack((miRNA, array))
        bottom = np.hstack((array.T, disease))
        self.symArray = np.vstack((top, bottom))

        self.startTime = time.time()
        self.originalHamilton = self.hamilton(self.symArray)

    def hamilton(self, array):
        """

        @return:
        """
        a = np.linalg.eigvalsh(array)
        values = []
        for k in range(3, self.numFeature + 3):
            wul = sum(a.real ** k)
            if wul > 0:
                values.append(math.log(wul))
            else:
                values.append(0.0)

        return values

    def calFeature(self, rowIndex, colIndex):
        """

        :param rowIndex:
        :param colIndex:
        :return:
        """

        if self.symArray[rowIndex][colIndex] == 1:
            self.symArray[rowIndex][colIndex] = 0
            self.symArray[colIndex][rowIndex] = 0

            values = self.hamilton(self.symArray)

            features = [self.originalHamilton[i] - values[i] for i in range(self.numFeature)]
            self.symArray[rowIndex][colIndex] = 1
            self.symArray[colIndex][rowIndex] = 1

        elif self.symArray[rowIndex][colIndex] == 0:
            self.symArray[rowIndex][colIndex] = 1
            self.symArray[colIndex][rowIndex] = 1

            values = self.hamilton(self.symArray)

            features = [values[i] - self.originalHamilton[i] for i in range(self.numFeature)]
            self.symArray[rowIndex][colIndex] = 0
            self.symArray[colIndex][rowIndex] = 0

        return features

    def calFeatures(self):
        """

        :return:
        """
        features = [[None for i in range(self.numCol)] for j in range(self.numRow)]
        count = 0
        for i in range(self.numRow):
            for j in range(self.numCol):
                features[i][j] = self.calFeature(i, self.numRow + j)
            count += 1
            print count, time.time() - self.startTime
        features = np.array(features)

        fileName = self.directory + "/feature" + str(self.numfold) + ".npy"
        print "writing"
        np.save(fileName, features)


class Interaction:
    """
    Class for storing instance values
    @author: Yadong Dong (yddong.gm@gmail.com)
    """
    def __init__(self, interactions = None):
        """

        :param interactions:
        """
        # referance to instances
        self.interactions = interactions

        # the values to be stored
        self.nodes = []

        # sample weight
        self.weight = 1.0

        # sample add information
        self.additionInfo = None

    def setInteractions(self, interactions):
        """

        @param interactions:
        @return:
        """
        self.interactions = interactions

    def setWeight(self, weight):
        """
        set instance weight
        @param weight: float
            instance weight
        """
        self.weight = weight

    def getWeight(self):
        """
         get instance weight

        @return:
            instance weight
        """
        return self.weight

    def copy(self):
        """
        Produces a shallow copy of this instance. he copy has access to the same instances.

        @return:
            instance
        """
        interaction = Interaction(self.interactions)
        interaction.nodes = self.nodes[:]
        interaction.weight = self.weight
        return interaction

    def setNodes(self, nodes):
        """

        @param nodes:
        @return:
        """
        if len(nodes) != 2:
            raise Exception("the length of interaction nodes should be two.\n")
        self.nodes = nodes[:]

    def __setitem__(self, index, value):
        if index < 0 or index >= len(self.nodes):
            raise Exception("the index is out of the interaction nodes range.\n")
        self.nodes[index] = value

    def __getitem__(self, index):
        if index < 0 or index >= len(self.nodes):
            raise Exception("the index is out of the interaction nodes range.\n")
        return self.nodes[index]

    def __str__(self):
        selfDescription = u''
        selfDescription += self.interactions.integerToNodes[self.nodes[0]]
        selfDescription += u'\t'
        selfDescription += self.interactions.integerToNodes[self.nodes[1]]
        if self.interactions.weighted:
            selfDescription += u'\t'
            selfDescription += str(self.weight)
        selfDescription += u'\n'
        return selfDescription


class Interactions:
    """
    Instances: handling an ordered set of weighted instances
    @author: Ya-Dong Dong (yddong.gm@gmail.com)
    """

    # relationship notation
    INTERACTIONSNAME = '@InteractionsName'

    # construction function
    def __init__(self, interactionsName=''):
        """

        @param interactionsName:
        """
        # relation name
        self.interactionsName = interactionsName

        # number of all instances
        self.numInteractions = 0

        # all Clusters list
        self.interactions = []

        self.weighted = False

        # the nodes of the clusters
        self.mapStart = 0
        self.nodes = set()
        self.nodesToInteger = {}
        self.integerToNodes = {}

        # nodes number of the clusters
        self.numNodes = 0

    def deepCopy(self):
        """
        copy samples data,not reference to the original sample set address.

        @return:
            new instances
        """
        interactions = Interactions(self.interactionsName)
        interactions.nodesToInteger.update(self.nodesToInteger)
        interactions.integerToNodes.update(self.integerToNodes)

        # copy interactions
        for interaction in self.interactions:
            interactionTmp = interaction(interactions)
            interactionTmp.setWeight(interaction.getWeight())
            interactionTmp.nodes = interaction.nodes[:]
            interactionTmp.numNodes = interaction.numNodes
            interactions.appendInteraction(interactionTmp)

        return interactions

    def copyStructure(self):
        """

        """
        interactions = Interactions(self.interactionsName)
        interactions.integerToNodes.update(self.integerToNodes)
        interactions.nodesToInteger.update(self.nodesToInteger)

        return interactions

    def getTotalWeights(self):
        """
        the sum of the total samples' weight

        @return:
            the sum of the total samples' weight
        """
        weights = 0.0
        for interaction in self.interactions:
            weights += interaction.getWeight()
        return weights

    def setInteractionsName(self, interactionsName):
        """
        set relationName

        @param interactionsName: string
            sample's relation name
        """
        self.interactionsName = interactionsName

    def appendInteraction(self, interaction):
        """
        append new cluster

        @param interaction: interaction
            new interaction
        """
        self.interactions.append(interaction)
        interaction.setInteractions(self)
        self.numInteractions += 1
        self.nodes.update(interaction.nodes)
        self.numNodes = len(self.nodes)

    def updateNodesIntegerMap(self, nodes):
        """

        """
        if isinstance(nodes, list):
            for node in nodes:
                if not self.nodesToInteger.has_key(node):
                    self.integerToNodes[len(self.nodesToInteger) + self.mapStart] = node
                    self.nodesToInteger[node] = len(self.nodesToInteger) + self.mapStart

        else:
            if not self.nodesToInteger.has_key(nodes):
                self.integerToNodes[len(self.nodesToInteger) + self.mapStart] = nodes
                self.nodesToInteger[nodes] = len(self.nodesToInteger) + self.mapStart

    def setNodesIntegerMap(self, nodesToInteger):
        """

        """
        self.nodesToInteger = nodesToInteger
        self.integerToNodes = {value:key for key,value in self.nodesToInteger.items()}

    def setMapStart(self, start):
        """
        """
        self.mapStart = start

    def setWeighted(self, flag):
        """

        @param flag:
        @return:
        """
        self.weighted = flag

    def toGraph(self):
        """

        """
        graph = nx.Graph()
        for interaction in self.interactions:
            if interaction[0] != interaction[1]:
                graph.add_edge(interaction[0], interaction[1], weight=interaction.getWeight())
        return graph

    def merge(self, interactions):
        """

        @param interactions:
        @return:
        """
        self.formGraph(interactions.toGraph())
        graph = self.toGraph()
        self.clear()
        self.formGraph(graph)

    def formGraph(self, graph):
        """
            get interactions from graph
        """
        edges = graph.edges(data='weight')
        for edge in edges:
            interaction = Interaction(self)
            interaction.setNodes(list(edge[:2]))
            interaction.setWeight(edge[2])
            self.appendInteraction(interaction)

    def dimensions(self):
        """
        get the length of the two dimensions
        @return:
        """
        dims = [set(), set()]
        for interaction in self.interactions:
            dims[0].add(interaction[0])
            dims[1].add(interaction[1])

        return [len(dim) for dim in dims]

    def toArrays(self):
        """

        @return:
        """
        array = np.zeros(self.dimensions())
        for interaction in self.interactions:
            array[interaction[0]][interaction[1]] = interaction.weight

        return array

    def splitTrainProbeArrayBasedOnCol(self):
        """

        @return:
        """
        numRow, numCol = self.dimensions()
        array = self.toArrays()

        arrays = [[array.copy(), np.zeros(self.dimensions())] for i in range(numCol)]

        for i in range(numCol):
            for j in range(numRow):
                arrays[i][0][j, i] = 0
                arrays[i][1][j, i] = array[j, i]

        return arrays

    def randomSplitTrainProbeArrary(self, numFold):
        """

        @param numFold:
        @return: arrays: the training and probe arrays for each fold.
        """
        randomIndex = range(self.numInteractions)
        random.seed(1)
        random.shuffle(randomIndex)

        edgesList = [[] for i in range(numFold)]
        for i in range(numFold):
            if i < numFold - 1:
                index = randomIndex[self.numInteractions / numFold * i: self.numInteractions / numFold * (i+1)]
            else:
                index = randomIndex[self.numInteractions / numFold * i:]

            for j in index:
                edge = [int(self.integerToNodes[self.interactions[j][0]]),
                        int(self.integerToNodes[self.interactions[j][1]]), self.interactions[j].weight]
                edgesList[i].append(edge)

        arrays = [[np.zeros(self.dimensions()), np.zeros(self.dimensions())] for i in range(numFold)]
        for i in range(numFold):

            for j in range(numFold):
                if i == j:
                    for edge in edgesList[j]:
                        arrays[i][1][edge[0] - 1][edge[1] - 1] = edge[2]
                else:
                    for edge in edgesList[j]:
                        arrays[i][0][edge[0] - 1][edge[1] - 1] = edge[2]

        return arrays

    def clear(self):
        """

        @return:
        """
        self.interactions = []
        self.numNodes = 0
        self.nodes = set()

    def __getitem__(self, item):
        if item < 0 or item >= len(self.interactions):
            raise Exception("the index is out of the clusters range.\n")
        return self.interactions[item]

    def __setitem__(self, key, value):
        if not isinstance(value, Interaction):
            raise Exception("parameter value must be the type:Interaction")
        if key < 0 or key >= self.numInteractions:
            raise Exception("the index is out of the interactions range.\n")
        self.interactions[key] = value

    def __len__(self):
        return len(self.interactions)

    def __str__(self):
        # str = u'@clustersName ' + self.interactionsName + u'\n'
        string = u''
        for interaction in self.interactions:
            string += interaction.__str__()
        return string

    def __delitem__(self, key):
        if key < 0 or key >= self.numInteractions:
            raise Exception('out of range')
        del self.interactions[key]
        self.numInteractions -= 1

    def __iter__(self):
        """

        """
        return iter(self.interactions)


class InteractionDataSource:
    """
    file: model for read and write Cluster data set

    @author: Ya-Dong Dong (yadong.gm@gmail.com)
    """
    CLUSTERS = 0
    PARTITION = 1

    def __init__(self, file_name, mapFile=None):
        """
        Construction function. Read cluster data

        @param file_name: string
            file name for data set
        """
        self.interactions = Interactions('')
        # read the map relation between nodes and integer

        # get structure from clusters clustersOrMapFile
        if isinstance(mapFile, Interactions):
            self.interactions = mapFile.copyStructure()

        elif isinstance(mapFile, dict):
            self.interactions.setNodesIntegerMap(mapFile)

        # get structure from file clustersOrMapFile
        elif isinstance(mapFile, str):
            self.mapFile_name = mapFile
            try:
                fileHandle = codecs.open(mapFile, 'r', 'UTF-8')
            except Exception as e:
                raise Exception("can not open the file: %s" % mapFile)

            self.curLine = 0
            self.content = []
            self.eliminateAnnotation(fileHandle)
            self.getMap()
            fileHandle.close()

        # read the interactions
        self.file_name = file_name
        # current line has been done
        self.curLine = 0
        self.content = []
        try:
            file_handle = codecs.open(file_name, 'r', 'UTF-8')
        except Exception as e:
            raise Exception("can not open the file: %s" % file_name)

        self.eliminateAnnotation(file_handle)
        # self.getInteractionsName()
        self.getInteractions()
        file_handle.close()

    def eliminateAnnotation(self, file):
        """
        eliminate annotations, and save the results to storage(content)

        @param file: FileHandel
            the file handle
        """
        for line in file:
            line = u"%s" % line
            if line.rfind(u'\n') != -1:
                line = line[0:len(line) - 1]  # eliminate \n
            pos = line.find(u'#', 0)
            if pos != -1:
                line = line[0:pos]

            line = line.replace(u'\t', u' ')
            line = line.strip()
            self.content.append(line)

    def findSplitPos(self, conList):
        """
        find the first pos of ' '
        @param conList:
            content
        @return:
            position
        """
        # find the split pos

        index0 = conList.find(u' ')  # english
        index1 = conList.find(u' ')  # chinese

        if index0 != -1 and index1 != -1:
            if index0 < index1:
                index = index0
            else:
                index = index1
        else:
            if index0 != -1:
                index = index0
            else:
                index = index1
        return index

    def getInteractionsName(self):
        """
        Sparse the structure of data set
        """
        content = self.content
        for curLine in range(len(content)):
            if not len(content[curLine]):
                continue
            conList = content[curLine]
            index = self.findSplitPos(conList)
            con = conList[0:index]
            con.strip()
            if self.isInteractionsName(con):
                con = conList[index:]
                self.interactions.setInteractionsName(con.strip())
                self.curLine = curLine + 1
            else:
                info = 'line ' + str(curLine) + ' should be interactions name' + self.file_name + u'\n'
                raise Exception(info)
            return

    def getInteractions(self):
        """
        Parse the instances of the data set
        """
        content = self.content
        for curLine in range(self.curLine, len(content)):
            if len(content[curLine]) == 0:
                continue
            if -1 != content[curLine].find(u',') or -1 != content[curLine].find(u'，'):
                conList = re.split(ur',|，', content[curLine])
            else:
                conList = re.split(ur' | ', content[curLine])

            while '' in conList:
                conList.remove(u'')

            for i in range(len(conList)):
                conList[i] = conList[i].strip()

            try:
                self.interactions.updateNodesIntegerMap(conList[0:2])
                interaction = Interaction(self.interactions)
                nodes = [self.interactions.nodesToInteger[conList[0]], self.interactions.nodesToInteger[conList[1]]]
                interaction.setNodes(nodes)
                if len(conList) == 3:
                    self.interactions.setWeighted(True)
                    interaction.setWeight(float(conList[2]))
                else:
                    self.interactions.setWeighted(True)
                    interaction.setWeight(1.0)
                self.interactions.appendInteraction(interaction)
            except Exception as e:
                e.exception += u" in line " + str(curLine)
                raise e

    def getMap(self):
        """
        Parse the map between nodes and integers
        """
        content = self.content
        nodesToInteger = {}
        for curLine in range(self.curLine, len(content)):
            if len(content[curLine]) == 0:
                continue
            if -1 != content[curLine].find(u',') or -1 != content[curLine].find(u'，'):
                conList = re.split(ur',|，', content[curLine])
            else:
                conList = re.split(ur' | ', content[curLine])

            while '' in conList:
                conList.remove(u'')

            for i in range(len(conList)):
                conList[i] = conList[i].strip()

            try:
                if conList[0].isdigit():
                    nodesToInteger[conList[1]] = int(conList[0])
                else:
                    nodesToInteger[conList[0]] = int(conList[1])
            except Exception as e:
                e.exception += u" in line " + str(curLine) + u"i: %s" % self.mapFile_name
                raise e
        self.interactions.setNodesIntegerMap(nodesToInteger)

    def isInteractionsName(self, key):
        """
        Check key is '@relation'
        @param key: string
            check the key
        @return:
            True if key == '@relation', False otherwise
        """
        key = key.strip()
        key = key.lower()
        att = Interactions.INTERACTIONSNAME.lower()

        if key == att:
            return True
        else:
            return False


class SimpleModel(object):

    def __init__(self, trainArray, probeArray, directory, numFeature, classifier=None):
        """
        Construction function. Init variables:
        """
        self.trainArray = trainArray
        self.probeArray = probeArray
        self.numRow, self.numCol = trainArray.shape

        self.numFeature = numFeature
        self.directory = directory
        self.classifier = classifier
        self.EP = None

    def readFeatures(self, num):
        """

        :param num:
        :return:
        """

        self.EP = np.load(self.directory + "/feature" + str(num) + ".npy")

    def getFeatures(self, rowIndex, colIndex):
        """

        @return:
        """
        features = []
        features.extend(self.EP[rowIndex][colIndex][0:self.numFeature])

        return features

    def toNumArray(self):
        """

        @return:
        """
        trainData = []
        testData = []
        trainLabel = []
        testLabel =[]
        for rowIndex in range(self.numRow):
            for colIndex in range(self.numCol):
                features = self.getFeatures(rowIndex, colIndex)
                trainData.append(features[:])
                trainLabel.append(self.trainArray[rowIndex][colIndex])
                if self.trainArray[rowIndex][colIndex] == 0:
                    testData.append(features[:])
                    testLabel.append(self.probeArray[rowIndex][colIndex])

        return np.array(trainData), np.array(testData), np.array(trainLabel), np.array(testLabel)


def calAUC(trainArray, probeArray, fold, queue1, queue2, method, numFeatures):
    """

    :param trainArray:
    :param probeArray:
    :param fold:
    :param queue1:
    :param queue2:
    :param method:
    :return:
    """
    numFeature = numFeatures
    numHidden = 20

    simpleModel = SimpleModel(trainArray, probeArray, directory + "features5FoldCV/", numFeature)
    # simpleModel = SimpleModel(trainArray, probeArray, directory + str(fold), classifier)
    simpleModel.readFeatures(fold)
    # simpleModel.buildClassifier()
    # aucValue = simpleModel.auc()
    # print(aucValue)

    X_train, X_test, y_train, y_test = simpleModel.toNumArray()
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)

    # lr = MLPRegressor(hidden_layer_sizes=(numHidden, numHidden, numHidden))
    # lr = LinearDiscriminantAnalysis(solver='lsqr')
    # lr = LogisticRegression()
    # lr = svm.SVR()
    if method is "SVR":
        lr = svm.SVR()
    elif method is "RF":
        lr = RandomForestRegressor()
    else:
        lr = MLPRegressor(hidden_layer_sizes=(numHidden, numHidden, numHidden))
    print "start training! " + str(fold)
    lr.fit(X_train_minmax, y_train)
    print "finished training" + str(fold)
    values = lr.predict(X_test_minmax)
    values = values.astype('float64')

    fpr, tpr, auc_thresholds = metrics.roc_curve(y_test, values)
    auc_score = metrics.auc(fpr, tpr)
    rocData = np.vstack((fpr, tpr))
    rocData = rocData.T
    np.savetxt(directory + "results5FoldCV/" + method + "/rocData" + str(fold) + ".csv", rocData, delimiter=',')

    precision1, recall, pr_threshods = metrics.precision_recall_curve(y_test, values)
    aupr_score = metrics.auc(recall, precision1)
    auprData = np.vstack((precision1, recall))
    auprData = auprData.T
    np.savetxt(directory + "results5FoldCV/" + method + "/auprData" + str(fold) + ".csv", auprData, delimiter=',')

    print "auc", auc_score, "aupr", aupr_score

    queue1.put(auc_score)
    queue2.put(aupr_score)


def calFeatures(trainArray, featurePath, numFold, numFeatures):
    """

    :param trainArray:
    :param featurePath:
    :param numFold:
    :param numFeatures:
    :return:
    """
    print "calculating features for fold: " + str(numFold)

    asymModel = AsymModel(trainArray, featurePath, numFold, numFeatures)
    asymModel.calFeatures()


if __name__ == '__main__':
	

    method = "MLP"
    if len(sys.argv) > 1:
    	method = sys.argv[1]
    
    numFold = 5
    numFeatures = 7
    directory = "data/"
    featuresPath = directory + "features5FoldCV"
    dataSource = InteractionDataSource(directory + "miRNA-disease.txt")
    interactions = dataSource.interactions

    arrays = interactions.randomSplitTrainProbeArrary(numFold)
    print len(os.listdir(featuresPath))

    if len(os.listdir(featuresPath)) is not numFold+1:
        print "calculating features"
        processCalList = []
        for i in range(numFold):
            process = Process(target=calFeatures, args=(arrays[i][0], arrays[i][1], i + 1, numFeatures))
            processCalList.append(process)
            process.start()
        for process in processCalList:
            process.join()

    queue1 = Queue(numFold)
    queue2 = Queue(numFold)
    processList = []
    for i in range(numFold):
        process = Process(target=calAUC, args=(arrays[i][0], arrays[i][1], i + 1, queue1, queue2, method, numFeatures))
        processList.append(process)
        process.start()
    for process in processList:
        process.join()

    aucList = [queue1.get() for i in range(numFold)]
    auprList = [queue2.get() for i in range(numFold)]
    print "average AUC", sum(aucList) / numFold
    print "average AUPR", sum(auprList) / numFold

    np.savetxt(directory + "results5FoldCV/" + method + "/aucList.txt", aucList)
    np.savetxt(directory + "results5FoldCV/" + method + "/auprList.txt", auprList)
