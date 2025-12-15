




class Level:
    def __init__(self):
        self.sstables = []
        self.count = 0
    
    def add_sstable(self,data):
        self.sstables.append(SSTable(data))

    def size(self):
        return len(self.sstables)


class SSTable:
    def __init__(self,data):
        self.data = data
    
class LSMTree:
    memtable = []
    levels = [Level() for _ in range(2)]

    
    def insert(self,value):
        if(len(self.memtable)  == 5):
            self.flush_memtable()
            self.memtable = [value]
        else:
            self.memtable.append(value)


    def flush_memtable(self,forced=False):
        self.levels[0].add_sstable(self.memtable)
        self.levels[0].count+=1
        if(forced or self.levels[0].count == 5):
            self.compactl0()
            self.levels[0].count = 0
        self.memtable = []

    def compactl0(self):
        for sstable in self.levels[0].sstables:
            self.levels[1].sstables.append(sstable)
        self.levels[0] = Level()


    def display_lsm(self):
        for level_index in range(len(self.levels)):
            for table in self.levels[level_index].sstables:
                print(table.data)