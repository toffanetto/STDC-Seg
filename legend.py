
import matplotlib.pyplot as plt
import json
import numpy as np


with open('./cityscapes_info.json', 'r') as fr:
        labels_info = json.load(fr)
lb_color = {el['trainId']: el['color'] for el in labels_info}
lb_name = {el['trainId']: el['name'] for el in labels_info}

out_image = np.zeros([19, 50, 100, 3], dtype=np.uint8)

plt.figure()

for label in range(19):
    out_image[label][:][:] = lb_color[label]
    plt.subplot(4,5,label+1)
    plt.title(lb_name[label], fontsize = 12)
    plt.imshow(out_image[label])
    plt.axis('off')
    
plt.suptitle("Colors per label")
plt.tight_layout(pad=0.5)
plt.savefig("./output/legend.pdf",bbox_inches='tight')
