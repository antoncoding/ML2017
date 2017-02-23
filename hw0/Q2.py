from PIL import Image
import sys

file_a = sys.argv[1]
file_b = sys.argv[2]

a = Image.open(file_a)
b = Image.open(file_b)
width = a.width
height = a.height

b = b.load()
a = a.load()
new_p = Image.new('RGBA', (width, height), color=(0,0,0,0))
canva = new_p.load()

for i in range(0,width):
    for j in range(0,height):
        if a[i,j] != b[i,j]:
            canva[i, j] = b[i, j]

new_p.save('ans_two.png')