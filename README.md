Final master thesis by Sergio Noé

# A deep learning approach for the quantification of molecular organization from single-molecule localization microscopy

## Abstract

Recent developments in super-resolution microscopy techniques have provided the necessary tools to observe single molecules in living cells. However, these techniques require improved methods for the quantitative analysis of the images.  The booming of deep learning offers a wide palette of approaches for image analysis that might be exploited for this task.
In this thesis, we developed a methods for the analysis of super-resolution images of exocytic sites. Exocytosis is a biological process necessary to release waste and other molecules in cells, and involves the targeted transportation and the fusion of vesicles with the plasma membrane.  We focused on the imaging of the exocyst complex, which mediates in the fusion of vesicles with the membrane of the cells. We developed a tool capable of counting the number of exocysts and determine the underlying structure on which exocysts could be located starting from super-resolution images. The approach relies on deep learning techniques to analyze and classify the images in different categories. To build this structure we take advantage of a widely use architecture, called U-Net, to generate two extra images from the microscopy image and then use the three of them to extract further information through two extra branches of the network, providing the number of exocysts present in the initial image and the structure of these exocysts. This composite structure provide a powerful method to extract relevant biological information from exocytosis imaging and provide the researchers with a new tool to further develop their studies.

## Structure

![Final structure](https://github.com/SergioNoe/Exocysts-analysis-with-AI/blob/main/images/Structure.jpeg)

## Setting up the environment

The environment needed to execute the python files can be found in anaconda.org with the link https://anaconda.org/sergio_noe/deepo

Us ethe following lines of code in the anaconda prompt to download and use the deepo environment.

```bash
conda env create sergio_noe/deepo
source activate deepo
```

## Acknowledgments

I would like to thank Dr. Carlo Manzo for help and guide me through all the thesis and the members of the QuBI lab at UVic-UCC for accept me on the group and help me when i needed during two full years. I would also like to thank Oriol Gallego and Marta Puig of the Gallego lab of the Department of Experimental and Health Sciences (DCEXS) at the Pompeu Fabra University, for stimulating discussions and for providing the experimental data used to test our network structure. Furthermore, I would like to thank Giovanni Volpe and thank Jes\'{u}s Pineda of the Soft Matter Lab at the University of Gothenburg, for providing insights and valuable suggestions for the design of the network architecture. The QuBI lab acknowledges the support of NVIDIA Corporation with the donation of the Titan Xp GPU used in this study.
