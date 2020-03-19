import glob
import os
import shutil
#files = os.system("ls")
    #This will list all the files in present #working directory

files = glob.glob('./*.bag')


for file in files:
    #print(str(file).rstrip('.bag').lstrip('./'))
    os.mkdir(str(file).rstrip('.bag').lstrip('./'))
    shutil.move(file,str(file).rstrip('.bag').lstrip('./'))

directory_to_check =  os.getcwd()
directories = [os.path.abspath(x[0]) for x in os.walk(directory_to_check)]
directories.remove(os.path.abspath(directory_to_check))

for i in directories:
      os.chdir(i)         # Change working Directory
      f = glob.glob('./*.bag')
      f=str(f[0]).lstrip('./')
      print(f)
      os.system('rs-convert -i '+f+' -p ./images -v ./csv')
    #os.system(mkdir str(file).rstrip('.bag').lstrip('./'))
    #print(files)
