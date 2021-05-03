import glob
import os
import shutil
import xml.etree.cElementTree as ET

home = '/home/jupyter'
root1 = 'VOCdevkit'
root2 = 'VOC2012'
root3 = 'ImageSets'

# ImageSets/Mainディレクトリの中身を空にする。
image_dir = os.path.join(home, root1, root2, root3, 'Main')
shutil.rmtree(image_dir)
os.mkdir(image_dir)

# Annotationsディレクトリ内のすべてのXMLファイルのパスを取得
voc_dir = os.path.join(home, root1, root2, 'Annotations','*.xml')
all_xml_pass = glob.glob(voc_dir)

train_data = []

for each_xml_pass in all_xml_pass:
    
    # ファイル名のみ取り出し
    xml_filename = os.path.basename(each_xml_pass)
    
    # XMLファイルを開きルートに移動
    tree = ET.parse(each_xml_pass)
    root = tree.getroot()

    # 'head'のバウンディングボックスを取得する。
    bndbox_all = []
    for child in root.findall('object/part'):
        bndbox = []
        if child.find('name').text == 'head':
            bndbox.append(child.find('bndbox/xmin').text)
            bndbox.append(child.find('bndbox/ymin').text)
            bndbox.append(child.find('bndbox/xmax').text)
            bndbox.append(child.find('bndbox/ymax').text)
        if len(bndbox)>0:
            bndbox_all.append(bndbox)
    
    # 'head'のデータのみを学習データとして出力する。
    if len(bndbox_all)>0:

        # 拡張子を除去したファイル名をリストに登録する。
        train_data.append(xml_filename.replace('.xml', ''))
        
        # JPEGファイル名を作成する。
        jpg_filename = xml_filename.replace('xml', 'jpg')

        # 新しいXMLファイルを作成する。
        new_root = ET.Element('annotation')
        
        # JPEGファイル名をセットする。
        ET.SubElement(new_root, 'filename').text = jpg_filename
        
        # width, height, depthを元のXMLからコピーする。
        Size = ET.SubElement(new_root, 'size')
        ET.SubElement(Size, 'width').text = root.find('size/width').text
        ET.SubElement(Size, 'height').text = root.find('size/height').text
        ET.SubElement(Size, 'depth').text = root.find('size/depth').text

        # バウンディングボックスをセットする。
        for new_budbox in bndbox_all:
            Object = ET.SubElement(new_root, 'object')
            
            ET.SubElement(Object, 'name').text = 'head'
            ET.SubElement(Object, 'difficult').text = '0'

            Bndbox = ET.SubElement(Object, 'bndbox')
            ET.SubElement(Bndbox, 'xmin').text = new_budbox[0]
            ET.SubElement(Bndbox, 'ymin').text = new_budbox[1]
            ET.SubElement(Bndbox, 'xmax').text = new_budbox[2]
            ET.SubElement(Bndbox, 'ymax').text = new_budbox[3]

        # 新しいXMLファイルを出力する。
        new_tree = ET.ElementTree(new_root) 
        new_xml = os.path.join(home, root1, root2, 'Annotations', xml_filename)
        new_tree.write(new_xml)

# ImageSets/Mainディレクトリに新しい'train.txt'を出力する。
text = "\n".join(train_data)
train_text = os.path.join(home, root1, root2, root3, 'Main', 'train.txt')
with open(train_text, "w") as f:
    f.write(text)

