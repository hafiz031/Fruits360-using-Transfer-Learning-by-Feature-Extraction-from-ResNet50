# introduce the directory where to prune
# IMPORTANT: it will DELETE files from the given directory to PRUNE it (so keep backup of that folder if required)

keep_frequency = 10 # keep one out of each 10s

import os
for root1, dirs1, files1 in os.walk(r"./Train", topdown=False): # It will delete images from this folder maintaining the interval
   for name1 in dirs1:
      subfolder = os.path.join(root1, name1)
      for root2, dirs2, files2 in os.walk(subfolder, topdown=False):
#          print(subfolder)
#          image_dir = os.path.join(root2, files2)
#          print(files2)# random sampling koro
          
          image_dir = []
          for i, image_name in enumerate(files2):
              image_dir.append(os.path.join(root2, image_name))
              
              # deleting files that are out of frequency
              if i % keep_frequency:
                  os.remove(image_dir[-1])
