"""
This script gets the midline in CCF space using a structure known to be in the middle of the 2
hemispheres
"""
import ants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    annotations = ants.image_read('/Users/adam.amster/Downloads/annotations_compressed_25.nii.gz').numpy()

    third_ventricle = np.where(annotations == 129)
    third_ventricle_ml_axis = third_ventricle[-1]
    fig, ax = plt.subplots()
    third_ventricle_ml_axis = pd.Series(third_ventricle_ml_axis)
    print(third_ventricle_ml_axis.describe())
    third_ventricle_ml_axis.plot.hist(ax=ax)
    fig.savefig('/tmp/plot.png')

    pd.DataFrame({'AP': third_ventricle[0], 'DV': third_ventricle[1], 'ML': third_ventricle[2]}).to_csv('/tmp/third_ventricle.csv', index=False)



if __name__ == '__main__':
    main()