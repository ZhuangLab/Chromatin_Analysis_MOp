
- Python scritps to decode and/or preprocess MERFISH data

 - **1_rna_merfish** includes scripts to preprocess RNA-MERFISH data:
   - RNA-MERFISH data is already decoded from raw images by [MERlin](https://github.com/ZhuangLab/MERlin).


 - **2_dna_merfish** includes scripts to decode and preprocess DNA-MERFISH data:
   - DNA-MERFISH data is to be decoded by [ImageAnalysis3](https://github.com/zhengpuas47/ImageAnalysis3) or the Zhuang lab archived [source_tools](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/tree/master/sequential_tracing/source) using the scripts describled in the folder.


- **3_data_to_4dn** includes scripts to convert RNA-MERFISH and DNA-MERFISH data to 4DN format:
   - MERFISH data for 4DN format is available at 4DN Data Portal at <https://data.4dnucleome.org/>, which can be accessed by accession numbers 4DNESPE924IP and 4DNESMTNNB3N.



- Note that downstream postanalysis [postanalysis](../postanalysis/README.md) described in this repository mainly uses the output formated from **1_rna_merfish** and **2_dna_merfish**, instead of the output formated from **3_data_to_4dn**



- To analyze the data downloaded from 4DN Data Portal, scritps are needed to convert the 4DN format back to the original input (see scritps described in **3_data_to_4dn** for details).

