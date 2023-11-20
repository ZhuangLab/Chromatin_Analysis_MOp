## DNA MERFISH preprocessing ##

This folder contains the template scripts to preprocess DNA MERFISH images to chromatin traces in Mecp2 +/- mice.
The template scripts are used to analyze the data whose RNA MERFISH was done on 20220809. This replicate corresponds to replicate B2 in the 4DN deposit.

Step 1 - Spot finding
    This script aims at identifying individual DNA spots by FISH signals at sub-pixel resolution at each field of view, and the coordinates of the spots are aligned.
    The inputs are dax files which are images files DNA MERFISH.
    The outputs are hdf5 files which contain the spot information as well as information regarding DNA DAPI images for segmentation.

Step 2 - Alignment
    This step aligns the DAPI from DNA MERFISH data to the DAPI of RNA MERFISH data. The cell segmentation from RNA MERFISH was then used for DNA MERFISH analysis.

Step 3 - Partition spots
    This steps partition all identified spots into respective cell segments.

Step 4 - Decoding
    This steps decodes all candidate spots within each cell based on the MERFISH codebook to their respective DNA loci.
    For replicates B3 and B4, no CTP12 library spots were decoded and those loci are decoded with the new CTP13 codebook.
    Codebooks are provided in the folder within this directory.

Step 5 - Export for picking
    This step simply exports the decoded spots into a file format for the next step - picking decoded spots into chromosome fibers

Step 6 - Picking
    By using the algorithm jie (modified from the original version and included as a submodule in the parent repository), this steps identifies possible chromosome fibers from decoded genomic loci.

The scripts from the next steps (Step 7 and onward) are used to analyze data from all four replicates

Step 7 - Sort spots
    In jie, there are potential mis-sorted genomic loci to homologous fibers. This script aims at resolving this issue.

Step 8 - Convert
    This script generates a dataframe and a list of all chromatin traces from all 4 replicates from Mecp2 +/- mice for further analysis.