from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from numpy import asarray
import os

import random

def make_square(im, min_size=32, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def convertIMG(nAnimali,nCartelle,label,risoluzione,path,pathCartelle):


    listImg=""
    daat= open("../../data/image.txt", "w")
    pp=[]
    for label in range(int(nCartelle)):
        j=0
        for filename in os.listdir(path+"/"+pathCartelle[label+1]):
            if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
                image=make_square(Image.open(str(path)+"/"+pathCartelle[label+1]+"/"+str(filename)),fill_color=(255,255,255,0)).convert('L')
                #ImageEnhance.Contrast(img).convert('L')).enhance(5).resize((risoluzione,risoluzione))
                parzIMG=[]

                parzIMG.append(image)
                w,h=image.size
                parzIMG.append(image.crop((w/4,h/4,w-(w/4),h-(h/4))))
                parzIMG.append(image.crop((0, 0, w/2, h/2)))
                parzIMG.append(image.crop((0, h/2, w / 2, h )))
                parzIMG.append(image.crop((w/2, 0, w , h / 2)))
                parzIMG.append(image.crop((w/2, h/2, w, h)))

                #parzIMG.append(ImageEnhance.Brightness(image).enhance(1.5))

                # parzIMG.append(ImageEnhance.Brightness(imm).enhance(1.5))
                imm=ImageOps.mirror(image)
                parzIMG.append(imm)


                parzIMG.append(imm.crop((w / 4, h / 4, w - (w / 4), h - (h / 4))))
                parzIMG.append(imm.crop((0, 0, w / 2, h / 2)))
                parzIMG.append(imm.crop((0, h / 2, w / 2, h)))
                parzIMG.append(imm.crop((w / 2, 0, w, h / 2)))
                parzIMG.append(imm.crop((w / 2, h / 2, w, h)))

                f=len(parzIMG)
                #for x in range(f):
                    #parzIMG.append(parzIMG[x].filter(ImageFilter.GaussianBlur(radius=5)))
                    #parzIMG.append(parzIMG[x].filter(ImageFilter.MinFilter(size=5)))

                for z in parzIMG:
                    data=asarray(z.resize((risoluzione,risoluzione)))
                    #data=data.astype('float32')
                    #data/=255
                    #data=data-data.mean()
                    #data= (data-data.mean()) /data.std()
                    currentIMG = str(j)+" "+str(label+1)

                    for d in data:
                        for k in d:
                                currentIMG+=" "+str(k)

                    pp.append(currentIMG)
                #listImg+=currentIMG+"\n"
                j+=1
            if j>=int(nAnimali):
                break
    while pp:
        if len(pp)==1:
            daat.write(pp.pop(0)+ "\n")
            break
        r=random.randint(0,len(pp)-1)
        daat.write(pp.pop(r)+ "\n")

    return daat


def takeImgNP(path,risoluzione):
    parzIMG = []
    np=[]
    image = make_square(Image.open(str(path))).convert('L')


    parzIMG.append(image)
    w, h = image.size
    parzIMG.append(image.crop((w / 4, h / 4, w - (w / 4), h - (h / 4))))
    parzIMG.append(image.crop((0, 0, w / 2, h / 2)))
    parzIMG.append(image.crop((0, h / 2, w / 2, h)))
    parzIMG.append(image.crop((w / 2, 0, w, h / 2)))
    parzIMG.append(image.crop((w / 2, h / 2, w, h)))

    imm = ImageOps.mirror(image)
    parzIMG.append(imm)

    parzIMG.append(imm.crop((w / 4, h / 4, w - (w / 4), h - (h / 4))))
    parzIMG.append(imm.crop((0, 0, w / 2, h / 2)))
    parzIMG.append(imm.crop((0, h / 2, w / 2, h)))
    parzIMG.append(imm.crop((w / 2, 0, w, h / 2)))
    parzIMG.append(imm.crop((w / 2, h / 2, w, h)))


    for z in parzIMG:
        parz=[]
        data = asarray(z.resize((risoluzione, risoluzione)))
        for d in data:
            for k in d:
                parz.append(k)
        np.append(parz)

    return np

