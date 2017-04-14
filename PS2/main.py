import ID3, parse, random
import matplotlib.pyplot as plt
import unit_tests as test

def noPruneVsPrune(file):
    wPrune = []
    nPrune = []
    sampleSize = []
    for i in range(10,301,10):
        print i
        (withPrune, noPrune) = testPruningOnHouseData(file, i)
        sampleSize.append(i)
        wPrune.append(withPrune)
        nPrune.append(noPrune)
    plt.title('With and Without Pruning')
    plt.plot(sampleSize, nPrune, label = "No Pruning")
    plt.plot(sampleSize, wPrune, label = "With Pruning")
    plt.xlabel("Number of training examples")
    plt.ylabel("Accuracy on testing set")
    plt.legend()
    plt.show()


def testPruningOnHouseData(inFile, trainSample):
    withPruning = []
    withoutPruning = []
    data = parse.parse(inFile)
    for i in range(100):
        random.shuffle(data)
        validTestCut = int(((len(data) - trainSample)/2) + trainSample)
        train = data[:trainSample]
        valid = data[:(trainSample*3)/10]
        test = data[trainSample:]

        tree = ID3.ID3(train, 'democrat')
        acc = ID3.test(tree, train)
        # print "training accuracy: ", acc
        acc = ID3.test(tree, valid)
        # print "validation accuracy: ", acc
        acc = ID3.test(tree, test)
        # print "test accuracy: ", acc

        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        # print "pruned tree train accuracy: ", acc
        acc = ID3.test(tree, valid)
        # print "pruned tree validation accuracy: ", acc
        acc = ID3.test(tree, test)
        # print "pruned tree test accuracy: ", acc
        withPruning.append(acc)
        tree = ID3.ID3(train + valid, 'democrat')
        acc = ID3.test(tree, test)
        # print "no pruning test accuracy: ", acc
        withoutPruning.append(acc)
    # print withPruning
    # print withoutPruning
    # print "average with pruning", sum(withPruning) / len(withPruning), " without: ", sum(withoutPruning) / len(
    #     withoutPruning)
    return (sum(withPruning) / len(withPruning), sum(withoutPruning) / len(
        withoutPruning))


def main():
    test.testID3AndEvaluate()
    test.testID3AndTest()
    test.testPruning()
    noPruneVsPrune('/home/freelancer/Documents/EECS349/HW2/PS2/house_votes_84.data')

if __name__=="__main__":
    main()