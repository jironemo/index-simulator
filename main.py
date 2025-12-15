from models.lsm_tree import LSMTree


lsm_tree = LSMTree()

for i in range(10):
    lsm_tree.insert((f'key{i}', f'value{i}'))


print("What would you like to do?")
print("1. Force Flush")
print("2. Insert Item")
option = 0
while(option != 3):
    option = int(input("Choose an option:"))

    if(option == 1):
        lsm_tree.flush_memtable(forced=True)
    elif(option == 2):
        lsm_tree.insert(input("Enter what you want to insert(key,value):"))
    
    print("Resulting LSM Tree:")
    lsm_tree.display_lsm()

