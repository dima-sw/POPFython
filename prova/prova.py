class tt:


    def r(self):

        print("ciao")
    def t(self):
        print("oo")

    def s(self,fun="r"):
        getattr(self,fun)()


a=tt()
a.s()