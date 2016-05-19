import seaborn
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

class MusicClassifier:
    def __init__(self, arffs, model, classes, in_size=900, test_size=0.3):
        self.arffs = arffs
        self.classes = classes
        self.test_size = test_size

        self.x = []
        self.y = []
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

        self.in_size = in_size

        #Unique label for each input .arff file
        self.l = 0

        self.testX = None
        self.testY = None

        self.trained_model = None
        self.model = model()

        self.prep_stage = False
        self.train_stage = False

    def add_dataset(self, arff):
        if self.prep_stage is True:
            self.arffs = []
        self.arffs.append('genres/%s.arff'%arff)
        self.classes.append(arff.split('.')[0])

        return self

    def prepare(self):
        self.preprocess()
        self.train()

        return self

    """ Preprocessing """
    def preprocess(self):
        print self.arffs
        for arff in self.arffs:
            print "Loading arff file: %s"%arff

            f = open(arff).readlines()[70:self.in_size]
            f = map(lambda line: map(lambda num: float(num), line.split(",")[:-1]), f)

            self.x += f
            self.y += [self.l for i in range(0, len(f))]

            self.l += 1

        self.train_x, self.test_x, \
        self.train_y, self.test_y = cross_validation.train_test_split(self.x, self.y,
                                                                      test_size=self.test_size)
        self.prep_stage = True


    def get_test_data(self):
        return zip(self.test_x, [self.classes[y] for y in self.test_y])

    def arff_to_vector(self, arff):
        f = open(arff).readlines()[70:self.in_size]
        f = map(lambda line: map(lambda num: float(num), line.split(",")[:-1]), f)

        return f

    def train(self):
        if not self.prep_stage:
            print "The dataset must be preprocessed first"
            return

        print("Training with %d of %d instances"%(len(self.train_x), len(self.x)))

        self.trained_model = self.model.fit(self.train_x, self.train_y)
        self.train_stage = True

    def predict(self, vector):
        return self.classes[self.model.predict(vector)]

    """ Evaluation """
    def score(self):
        sc = self.model.score(self.test_x, self.test_y)
        return sc

    def bulk_predict(self, data):
        predicted = map(lambda record: self.model.predict(record), data)

        return predicted


def main():
    datasets = ['metal', 'hiphop', 'classical', 'jazz', 'disco', 'pop', 'reggae']

    clf = MusicClassifier([], LogisticRegression, [], in_size=1000, test_size=0.2, n_components=-1)

    """ Evaluation """
    def one_by_one():
        clf.add_dataset('rock')
        for ds in datasets:
            clf.add_dataset(ds)
            clf.prepare()
            print "%d, %f"%(len(clf.arffs), clf.score())

    def all():
        clf.add_dataset('rock')\
            .add_dataset('metal')\
            .add_dataset('hiphop')\
            .add_dataset('classical')\
            .add_dataset('jazz')\
            .add_dataset('disco')\
            .add_dataset('pop')\
            .add_dataset('reggae')\
            .prepare()

        test = clf.get_test_data()
        for i in range(0,20):
            x = test[i][0]
            y = test[i][1]

            #print (clf.predict(x), y)

    all()
    print clf.score()


if __name__ == "__main__":
    main()