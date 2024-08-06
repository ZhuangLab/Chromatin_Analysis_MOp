

Python scritps to for library design, image preprocessing, and postanalysis for MERFISH data of the primary motor cortex (MOp) from wild-type mouse brains.


Main components of this respository: 

 - **library_design** includes scripts for probe design for MERFISH imaging, 
 
   - Raw images from MERFISH imaging were then decoded and preprocessed by scripts in **preprocess**.


 - **preprocess** includes scripts to decode and/or preprocess RNA-MERFISH and DNA-MERFISH data,
 
   - Preprocessed data were then used to generates output summary for scripts in **postanalysis**.

 - **postanalysis** includes scripts to analyze DNA-MERFISH data for quantification and figures.

 - **postanalysis_2024** includes the updated (2024) scripts to analyze DNA-MERFISH data for quantification and figures.

 - **functions** includes scripts that are used for **postanalysis**.


 - **external** includes scripts that analyze external data, which are then used for **postanalysis**.


 - **experiment** includes scripts that were used for performing MERFISH imaging.



How to navigate this respository: 

 - Note that the scripts above were typically written in jupyter notebooks.

 - Such jupyter notebooks were used to run analysis locally with the required functions and inputs.

 - Such required functions and inputs were often generated from another related notebooks. 

 - Description of where to identify the related notebooks can be found either in the **README.md** file or in the notebook itself.

