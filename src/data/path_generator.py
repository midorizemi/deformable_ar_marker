"""
File path generator
"""
import os


class DataDir(object):
    def __init__(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__),  os.pardir, os.pardir, 'data'))
        self.pwd = self.root

    def dir_path(self, name):
        return os.path.join(self. root, name)

    def data(self):
        return self.dir_path('data')

    def features(self):
        return self.dir_path('features')

    def logs(self):
        return self.dir_path('logs')

    def raw(self):
        return self.dir_path('raw')


class DataSet(DataDir):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.pwd = os.path.join(self.raw(), name)

    def get_set(self):
        return list(map(filter, os.listdir(self.pwd)))

    def filter(self, file):
        return file


class TemplateData(DataDir):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.pwd = os.path.join(self.raw(), 'template', name)
        self.featuresd = os.path.join(self.features(), name)

    def get_template_image(self):
        return os.path.join(self.pwd, "{}.{}".format(self.name, 'png'))

    def get_feature_pickle(self, name):
        return os.path.join(self.featuresd, "{}.{}".format(name, 'pickle'))


if __name__ == '__main__':
    a = DataDir()
    b = DataSet('test')
    c = TemplateData('temp')
