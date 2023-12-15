
class DataWrapper:

    def __init__(self, args):
        self.args = args
    

    def get_data(self):
        
        if self.args.datatype=='arcene':
            from src.data.arcene import Arcene
            arcene=Arcene()
            return arcene.get_data()
        elif self.args.datatype=='twomoon':
            from src.data.twomoon import Twomoon_synthetic
            twomoon=Twomoon_synthetic(100,1000)
            return twomoon.create_data()
        elif self.args.datatype=='spambase':
            from src.data.spambase import Spam
            spam=Spam()
            return spam.load_data()

        else:
            raise NotImplementedError("This datatype is not implemented yet")

        pass