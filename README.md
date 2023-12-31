# Project 2

## Task: 
Large machine learning (esp. deep learning) models can be pre-trained on a large amount of diverse data (usually, unlabelled data), and then be customized for various downstream tasks. In this project, you will use a pre-trained "large" foundation model to classify if a patient has ARDS (acute respiratory distress syndrome) based on his clinical note. A small training set is provided for you to build a machine learning pipeline for this task. You may need to finetune the pre-trained model using the provided training set. You will need to apply your pipeline to the test set and save the results in "test_result.csv", in which each row is the prediction of the corresponding row in the test set.

## Datasets: 
(a) The training set file contains two columns. The first column is the notes, and the second column is the class label (TRUE or FALSE); (b) The test set file has only one column containing the notes. Each row in the training and test files represent a patient. Both files are .pkl files.

## Submit: 

[70 points] The prediction result file "test_result.csv". We will use the F1 score to evaluate the performance of your models. Please follow the format requirement specified above. Otherwise, points will be deducted depending on how much effort TAs need to parse your results. All submissions will be ranked into three groups (high, medium, and low) according to the test F1 scores. The high, medium, and low groups will receive 70, 65, and 60 points, respectively.

[30 points] Source code. The source code must contain extensive comments to elucidate the logic of the program.. 


Acute respiratory distress syndrome (ARDS) is a serious lung condition that causes low blood oxygen. In this project, you need to make a preidction about if the patient has ARDS based on their medical notes. Each patient may have multiple notes.

## The following are the instructions and project template:

Let's explore how to fine-tune an LLM on a single commodity GPU with Ludwig, an open-source package that empowers you to effortlessly build and train machine learning models like LLMs, neural networks and tree based models through declarative config files.

In this notebook, we'll show an example of how to fine-tune Llama-2-7b to make prediction for the ARDS dataset.

By the end of this example, you will have gained a comprehensive understanding of the following key aspects:

### Ludwig:
An intuitive toolkit that simplifies fine-tuning for open-source Language Model Models (LLMs).

### Exploring the base model with prompts: 
Dive into the intricacies of prompts and prompt templates, unlocking new dimensions in LLM interaction.

### Fine-Tuning Large Language Models:
Navigate the world of model fine-tuning optimizations for getting the most out of a single memory-contrained GPU, including: LoRA and 4-bit quantization.

## Goal: Use LLMs For Medical-Case Prediction 🏥

In this webinar, the goal is to use an LLM for prediction. The model will take natural language as input, and should return true(ARDS patient)/false(Non-ARDS patient) as output. We're first going to iterate on a base Llama-2-7b model with prompting, and finally instruction-fine-tune the model.

As an example, if we prompt the model with this instruction:

> Instruction: Based on the provided context, return true if the pation has ARDS, otherwise return false.

> Context: Note 1: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with hypoxemic respiratory ___\ntransferred from OSH  // please evaluate ETT and line position, please\nevaluate b/l chest tube, please evaluate etiology of hypoxemic respiratory\nfailure     please evaluate ETT and line position, please evaluate b/l chest\ntube, please evaluate etiology of hypoxemic respiratory failure\n\nIMPRESSION: \n\nIn comparison with the study of ___ from an outside facility, the the\nextensive pneumomediastinum is much less prominent.  There are improved lung\nvolumes.  Cardiac silhouette is within normal limits though there are diffuse\nareas of increased opacification involving both lungs.  Subsequent study\ndictated previously suggests that this appearance could reflect ARDS, though\npulmonary edema or widespread infection could be considered.\nBilateral chest tubes are in place and there are sub tiny apical\npneumothoraces on both sides.\n\nNote 2: INDICATION:  ___ year old woman with hypoxic respiratory failure  // ET tube\nplacement, pneumonthorax\n\nTECHNIQUE:  Chest PA and lateral\n\nCOMPARISON:  Chest radiograph from ___ from earlier today\n\nFINDINGS: \n\nThe lung volumes are stable.  Moderate to severe pulmonary edema is unchanged.\nThe cardiac silhouette is stable.  There is interval development of\npneumomediastinum with air tracking superiorly into the neck and right\nsupraclavicular region.  There is also some air surrounding the aortic arch. \nStable calcification of the aortic arch.  Any residual right apical\npneumothorax is tiny, if any.  Bilateral chest tubes are intact.  The ETT\nterminates abruptly 4.3 cm from the carina.\n\nIMPRESSION: \n\nInterval development of pneumomediastinum.  Unchanged moderate-severe\npulmonary edema.\n\nNOTIFICATION:   The findings were discussed with ___, M.D. by ___\n___, M.D. on the telephone on ___ at 2:02 ___, 5 minutes after\ndiscovery of the findings.\n\nNote 3: EXAMINATION:  CT CHEST W/O CONTRAST\n\nINDICATION:  ___ year old woman with hypoxic respiratory failure  //\ncharacterization of infilrates\n\nTECHNIQUE:   Multidetector CT performed without the administration of contrast\nof the entire volume of the thorax with multi planar reformations and MIP\nreconstructions.\n\nDOSE:  Acquisition sequence:\n   1) Spiral Acquisition 5.1 s, 32.9 cm; CTDIvol = 5.8 mGy (Body) DLP = 188.6\nmGy-cm.\n Total DLP (Body) = 189 mGy-cm.\n\nCOMPARISON:  ___\n\nFINDINGS: \n\nFINDINGS:\n\nNECK, THORACIC INLET, AXILLAE, CHEST WALL: No thyroid lesions.  No\nsupraclavicular or axillary adenopathy.  No gross breast lesions.  Right-sided\nPICC line in situ terminating in the right axillary vein (3, 9).  Moderate\nsubcutaneous air in the chest wall.\n\nUPPER ABDOMEN: This study was not tailored to evaluate the subdiaphragmatic\norgans.  Feeding tube in situ in the stomach.  The adrenals appear normal. \nHypodense cystic lesion in the midpole of the right kidney measuring 18 mm in\ndiameter with a slightly coarse mural calcification and is incompletely imaged\nand further characterization with ultrasound is advised.\n\nMEDIASTINUM: Moderate pneumomediastinum. Subcentimeter mediastinal lymph\nnodes.\n\nHILA: No hilar adenopathy.\n\nHEART and PERICARDIUM: Normal cardiac configuration.  Relative hypodensity of\nthe blood pool suggesting anemia.  No aortic valve or coronary artery\ncalcifications.\nPLEURA: Bilateral chest tubes in situ.  The left tube is kinked as it enters\nthe left pleural space (3, 31) and its function should be correlated\nclinically.  No significant residual pneumothorax.\nLUNG:\n\n-PARENCHYMA:  There is a diffuse interstitial pattern with ground-glass\nopacification of the lungs with apical basal and posterior gradient as\nevidenced by mild ground-glass opacification in the anterior aspect of the\nlungs, moderate ground-glass opacification of the mid lung zones and severe\nground-glass and consolidation seen in the posterior basal aspect of the\nlungs.  A few indeterminate pulmonary nodules.\n-AIRWAYS:  Endotracheal tube in-situ with the tip 23 mm proximal to the\ncarina.  The airways are patent to the subsegmental and.  Mild, but varicoid\nbronchiectasis most pronounced in the lower lobes.\n-VESSELS:  The pulmonary arteries not enlarged.\nCHEST CAGE: Spondylotic changes of the thoracic spine.  No lytic/ destructive\nbony lesions.\n\nIMPRESSION: \n\nImaging findings in keeping with acute lung injury/ ARDS (diffuse alveolar\ndamage) transitioning between the acute/exudative phase to the organizing\nphase.\nThe posterior basal consolidation most likely reflects a combination of\nexudate and atelectasis, but please note that infection cannot be excluded\nwith certainty.\n\nModerate pneumomediastinum, but no features of tension.\n\nRight-sided PICC line in situ terminating in the right axillary vein.\n\nThe left-sided chest tube is kinked as it enters the pleural space and its\nfunction should be correlated clinically.\n\nRECOMMENDATION(S):  Ultrasound of the right kidney.\n\nNote 4: INDICATION:  ___ with PMH of DM, HLD who presents as a transfer from OSH\nwith acute hypoxemic hypercarbic respiratory failure concerning for ARDS, s/p\nsubclavian line placement.  // please eval placement of new L subclavian line \nContact name: ___: ___\n\nTECHNIQUE:  Chest PA and lateral\n\nCOMPARISON:  Chest radiograph from ___ earlier today\n\nFINDINGS: \n\nThe left central subclavian catheter terminates in the cavoatrial junction. \nThe lung volumes are stable.  Moderate to severe pulmonary edema is unchanged.\nThe cardiac silhouette is stable.  Slight interval improvement\npneumomediastinum, however the neck is beyond the margins of this image. \nStable calcification of the aortic arch.  Bilateral chest tubes are intact. \nThe ETT terminates approximately 3 cm from the carina.  The NG tube traverses\nthe diaphragm however the tip is not visualized on this image.\n\nIMPRESSION: \n\nLeft central subclavian catheter terminates in the cavoatrial junction. \nInterval improvement of pneumomediastinum.\n\nNote 5: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with ARDS  // eval interval change in pulm\ninfiltrates      eval interval change in pulm infiltrates\n\nIMPRESSION: \n\nCompared to chest radiographs ___ through ___.\n\nBibasilar consolidation is more pronounced.  Ground-glass opacification in mid\nand upper lung zones is stable.  Pleural effusions are small.  Right apical\npneumothorax is tiny.  Heart size is normal.\n\nET tube, left subclavian line, right upper thoracostomy tube all in standard\nplacements unchanged.  Right axillary catheter ends outside the chest.  Left\npleural drainage catheter is oriented horizontally.\n\nNote 6: EXAMINATION:  RENAL U.S.\n\nINDICATION:  ___ with PMH of DM, HLD who presents as a transfer from OSH\nwith acute hypoxemic hypercarbic respiratory failure concerning for ARDS  //\nplease eval R renal cyst for concerning features, e.g. RCC\n\nTECHNIQUE:  Grey scale and color Doppler ultrasound images of the kidneys were\nobtained.\n\nCOMPARISON:  None.\n\nFINDINGS: \n\nThe right kidney measures 12.0 cm. The left kidney measures 11.8 cm.  A simple\ncyst is seen in the lateral left kidney measuring 1.6 x 0.9 x 1.2 cm.  There\nare 2 adjacent cysts in the mid right kidney.  1 of the is simple, and\nmeasures 1.4 x 1.8 x 1.5 cm.  The other cyst, which contains internal echoes\nand rim calcification, measures 2.1 x 1.9 x 1.7 cm.  This does not demonstrate\na solid component or internal vascularity.  Normal cortical echogenicity and\ncorticomedullary differentiation are seen bilaterally.\n\nA Foley seen within the decompressed bladder.\n\nIMPRESSION: \n\n1.  Bilateral simple renal cysts.\n2.  Complex right renal cyst with a coarse calcification.  No solid renal\nmasses identified.\n\nNote 7: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with ards, bilateral chest tubes  // any\ninterval change in lungs? L chest tube clamped      any interval change in\nlungs? L chest tube clamped\n\nIMPRESSION: \n\nIn comparison with the study of earlier in this date, with the left chest tube\nclamped there is no evidence of enlargement of the tiny apical pneumothorax. \nSmall amount of subcutaneous gas is seen along the left lateral chest wall.\nThe diffuse bilateral pulmonary opacifications are slightly less prominent,\nmost likely reflecting the better inspiration of the patient.\n\nNote 8: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with ARDS, bilateral pneumothoraces  //\nprogression of infiltrates, PTX with bilateral chest tubes      progression of\ninfiltrates, intubated\n\nIMPRESSION: \n\nComparison to ___.  The monitoring and support devices, including\nthe left chest tube, are in stable position.  The medial aspect of the left\npneumothorax has increased in size.  There is no evidence of tension.  The\nvery widespread parenchymal opacities have also minimally increased, notably\nat the right and left lung bases.  No other changes are noted.  The right\nchest tube is in stable position.\n\nNote 9: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with h/o ARDS and b/l PTX s/p removal of L\nchest tube  // please evaluate for reaccumulation of PTX s/p chest tube\nremoval on left\n\nTECHNIQUE:  Chest single view\n\nCOMPARISON:  ___ 05:02\n\nFINDINGS: \n\nThe apical pneumothorax stable.  Pneumomediastinum has decreased in size. \nSmall left apical pneumothorax is similar to minimally more prominent. \nAppliances are in good position.  Normal heart size, pulmonary vascularity. \nDecreased bilateral hazy pulmonary opacities.  Improved bibasilar\nconsolidations.  Small volume subcutaneous emphysema right neck base.\n\nIMPRESSION: \n\nDecrease pneumomediastinum.  Small left apical pneumothorax, similar to\nminimally increased.  Stable tiny right apical pneumothorax.  Improved lung\nparenchymal findings.\n\nNote 10: INDICATION:  ___ with PMH of DM, HLD who presents as a transfer from OSH\nwith acute hypoxemic hypercarbic respiratory failure concerning for ARDS //\nmonitor interval change\n\nTECHNIQUE:  Chest PA and lateral\n\nCOMPARISON:  ___.\n\nFINDINGS: \n\nEndotracheal tube in-situ with the tip at the level of the medial clavicles 58\nmm proximal to the carina.  Nasogastric tube in situ coursing out of sight\ninferiorly. Left-sided subclavian central catheter tip in the mid to distal\nSVC.  Right-sided chest drain in situ.  Small to moderate pneumomediastinum\nwith subcutaneous air also seen in the neck and bilateral pectoralis muscles. \nDiffuse pulmonary ground-glass opacification with mild consolidation in the\nlung bases are essentially unchanged in keeping with ARDS.\n\nIMPRESSION: \n\nAs above\n\nNOTIFICATION:   The findings were discussed with ___, M.D. by ___\n___, M.D. on the telephone on ___ at 6:00 ___, 20 minutes after\ndiscovery of the findings.\n\nNote 11: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with ards on vent with chest tubes now with\nincreasing pressure readings on esophageal balloon and subQ air  // any\ninterval change? any reaccumulation of pneumothorax/     any interval change?\nany reaccumulation of pneumothorax/\n\nIMPRESSION: \n\nIn comparison with the study of ___, there is little overall change. \nMonitoring and support devices are stable.  Moderate pneumomediastinum\npersists with gas extending into the neck and in the pectoral region\nbilaterally.  Diffuse opacifications bilaterally are essentially unchanged.\n\nNote 12: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with ARDS intubated with chest tubes and subQ\nair  // Eval for interval change      Eval for interval change\n\nIMPRESSION: \n\nET tube tip is 4 cm above the carinal.  NG tube tip is in the stomach.  Left\nsubclavian line tip is at the level of mid SVC.  Right chest tube is in place.\n\nPulmonary edema is substantial.  There extensive amount of subcutaneous air. \nNo definitive pneumothorax or pneumomediastinum currently seen.\n\nNote 13: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with ARDS, bilateral pneumothorax, subcutaneous\nemphysema, pneumo mediastinum  // Evaluate progression of PTX,\npneumomediastinum     Evaluate progression of PTX, pneumomediastinum\n\nIMPRESSION: \n\nCompared to chest radiographs ___ through ___:\n\nSubcutaneous emphysema in the chest wall and neck has improved since ___.  Residual pneumomediastinum is mild pneumothorax minimal if any at the\nright apex.  No appreciable pleural effusion.\n\nDiffuse ground-glass opacification in the lungs is improved since ___,\nbut not more recently.  There is the suggestion of developing bronchiectasis\nin bibasilar consolidation which could be due to developing fibrosis.\n\nHeart size normal.  Pleural effusion small if any.\n\nMultiple cardiopulmonary support devices in standard placements.\n\nNote 14: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with ARDS and pneumothorax s/p R chest tube  //\neval interval improvement of subcutaneous air, interstitial infiltrates     \neval interval improvement of subcutaneous air, interstitial infiltrates\n\nIMPRESSION: \n\nComparison to ___.  Stable monitoring and support devices.  In\nparticular, the right chest tube is in stable position.  No evidence of\npneumothorax.  Normal size of the heart.  Stable mild bilateral areas of\nbasilar atelectasis and mild fluid overload.  No pleural effusions.  Stable\nnormal appearance of the heart.\n\nNote 15: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ year old woman with ET tube  // assess tube placement     \nassess tube placement\n\nIMPRESSION: \n\nCompared to chest radiographs ___ through ___.\n\nSmall volume of pneumomediastinum and tiny right apical pneumothorax are are\nunchanged.  Subcutaneous emphysema has nearly resolved.\n\nGlobal ground-glass opacification is still present throughout the lungs along\nwith consolidation at the lung bases, improved slightly on the left.  Pleural\neffusion is small if any.  Heart size is normal.\n\nET tube, right apical thoracostomy tube, left subclavian line are in standard\nplacements and an esophageal drainage tube passes into the stomach and out of\nview.\n\nNote 16: EXAMINATION:  CHEST (PORTABLE AP)\n\nINDICATION:  ___ with PMH of DM, HLD who presents as a transfer from OSH\nwith acute hypoxemic hypercarbic respiratory failure concerning for ARDS with\nR chest tube  // r/o pneumothorax     r/o pneumothorax\n\nIMPRESSION: \n\nIn comparison with the study of earlier in this date, the right chest tube\nremains in place.  There is a tiny apical pneumothorax and a small amount of\npneumomediastinum.\nOtherwise little change.\n\n


We want the model to produce exactly this response:

> Response: true

## A Quick 2 Minute Introduction To Ludwig

Every Ludwig model is based on a config, which requires at least input feature and one output feature to be defined. For example,

> input_features:
> 
>  - name: instruction
>    
>    type: text
>    
>output_features:
> 
> - name: output
> - 
>    type: text

is a simple Ludwig config that tells Ludwig to use the column called instruction in our dataset as an input feature and the output column in our dataset as an output feature. This is the simplest Ludwig config we can define - it's just 6 lines and works out of the box!

To make Ludwig compatible with LLMs, Ludwig 0.8 introduced a new model_type called llm and a new keyword base_model that must be specified:

>model_type: llm
>
>base_model: meta-llama/Llama-2-7b-hf
>
>input_features:
>
>  - name: instruction
>    type: text
>    
> output_features:
>
> - name: output
>
> - type: text
    
    
The model_type parameter indicates is used to tell Ludwig you want to use the LLM model type (Ludwig supports LLMs, general deep neural networks and trees). The base_model parameter is the path to any HuggingFace CausalLM listed here.

For this webinar, we're going to make use of the Python LudwigModel API. This requires just one main object during initialization: a YAML config defining your training pipeline. The initialized LudwigModel object then exposes a variety of methods like preprocess(), train(), evaluate() and predict(). We will see this in practice in the next few sections.

## Basic Setup 🧰

We're going to install Ludwig, setup our HuggingFace Token and load our dataset that we will be running experiments with.

# Install Ludwig and Ludwig's LLM related dependencies.

Install Ludwig from the latest release

```
!pip uninstall -y tensorflow --quiet

!pip install ludwig

!pip install ludwig[llm]

Installing build dependencies ... done

  Getting requirements to build wheel ... done
  
  Preparing metadata (pyproject.toml) ... done
  
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.7/7.7 MB 22.9 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 60.6 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 682.2/682.2 kB 44.0 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 80.8/80.8 kB 10.7 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 49.4/49.4 kB 5.4 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 519.2/519.2 kB 48.4 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 280.2/280.2 kB 31.1 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98.1/98.1 kB 14.3 MB/s eta 0:00:00
     
  Installing build dependencies ... done
  
  Getting requirements to build wheel ... done
  
  Preparing metadata (pyproject.toml) ... done
  
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 232.0/232.0 kB 26.2 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 880.6/880.6 kB 49.8 MB/s eta 0:00:00
     
  Preparing metadata (setup.py) ... done
  
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 65.7 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.9/17.9 MB 79.6 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 92.5/92.5 MB 9.1 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0/100.0 kB 13.6 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 154.3/154.3 kB 22.0 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.5/57.5 kB 8.1 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 51.1/51.1 kB 7.6 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 268.8/268.8 kB 33.4 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 83.4 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.5/62.5 kB 9.1 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.4/58.4 kB 8.4 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98.7/98.7 kB 14.2 MB/s eta 0:00:00
     
  Building wheel for ludwig (pyproject.toml) ... done
  
  Building wheel for gpustat (pyproject.toml) ... done
  
  Building wheel for sacremoses (setup.py) ... done
  
DEPRECATION: git+https://github.com/ludwig-ai/ludwig.git@master#egg=ludwig[llm] contains an egg fragment with a non-PEP 508 name pip 25.0 will enforce this behaviour change. A possible replacement is to use the req @ url syntax, and remove the egg fragment. Discussion can be found at https://github.com/pypa/pip/issues/11617

  Installing build dependencies ... done
  
  Getting requirements to build wheel ... done
  
  Preparing metadata (pyproject.toml) ... done
  
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.0/86.0 kB 2.7 MB/s eta 0:00:00
     
  Preparing metadata (setup.py) ... done
  
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.6/17.6 MB 24.7 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 258.1/258.1 kB 24.8 MB/s eta 0:00:00
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.6/85.6 kB 11.9 MB/s eta 0:00:00
     
  Building wheel for sentence-transformers (setup.py) ... done
  ```
  
Enable text wrapping so we don't have to scroll horizontally and create a function to flush CUDA cache.

  ```
  from IPython.display import HTML, display

def set_css():

  display(HTML('''
  
  <style>
    
    pre {
      
        white-space: pre-wrap;
      
    }
    
  </style>
  
  '''))

get_ipython().events.register('pre_run_cell', set_css)

def clear_cache():

  if torch.cuda.is_available():
  
    model = None
    
    torch.cuda.empty_cache()
    
  ```

# Setup Your HuggingFace Token 🤗
We'll be exploring Llama-2 today, which a model released by Meta. However, the model is not openly-accessible and requires requesting for access (assigned to your HuggingFace READ token).

Obtain a HuggingFace API Token and request access to Llama2-7b-hf before proceeding. You may need to signup on HuggingFace if you don't aleady have an account: https://huggingface.co/join

Incase you haven't been given access to Llama-2-7b, that is alright. We can just use Llama-1 for the rest of this example: huggyllama/llama-7b.

```
import getpass

import locale; locale.getpreferredencoding = lambda: "UTF-8"

import logging

import os

import torch

import yaml

from ludwig.api import LudwigModel


os.environ["HUGGING_FACE_HUB_TOKEN"] = getpass.getpass("Input Your Huggingface READ Token:")

assert os.environ["HUGGING_FACE_HUB_TOKEN"]
```

### Before you run the next cells, please manually upload the data file to google drive.

Mount google drive to colab so we can access the dataset.

```
from google.colab import drive
drive.mount('/content/drive')

```

## Import The ARDS Dataset 📋

If you can't load the dataset, please check if you have mounted google drive, and if the file name/path is correct.

```
from google.colab import data_table; data_table.enable_dataframe_formatter()

import numpy as np; np.random.seed(123)

import pandas as pd

import pickle as pkl

df = pd.read_pickle("/content/drive/MyDrive/project2_train.pkl")

df = df.fillna("")

# We're going to create a new column called `split` where:

# 90% will be assigned a value of 0 -> train set

# 10% will be assigned a value of 1 -> validation set

# Calculate the number of rows for each split value

total_rows = len(df)

split_0_count = int(total_rows * 0.9)

split_1_count = total_rows - split_0_count


# Create an array with split values based on the counts

split_values = np.concatenate([

    np.zeros(split_0_count),

    np.ones(split_1_count),

])

# Shuffle the array to ensure randomness

np.random.shuffle(split_values)

# Add the 'split' column to the DataFrame

df['split'] = split_values

df['split'] = df['split'].astype(int)
```

## Understanding The ARDS Dataset

```
df.head(2)
```

| index | text | label | split|
|---|---|---| --- |
| 0 | Note 1: ADDENDUM: In the impression, it states that the free fluid within the rectovesicular space with the layering hematocrit could be a possible bowel perforation. This is incorrect. This fluid collection in the pelvis represents an intraperitoneal extension of the right retroperitoneal hematoma and is not suggestive of bowel perforation. Note 2: REASON FOR EXAMINATION: Hypoxemia. Portable AP chest radiograph was reviewed in comparison to ___. There is interval progression of left lower lobe retrocardiac consolidation currently obscuring the entire left lower lobe and the hemidiaphragm. There is also progression of the right basal consolidation and new right pleural effusion demonstrated. Upper lungs are essentially unchanged. Replaced mitral valve projecting over the significantly calcified mitral annulus is redemonstrated. The right PICC line tip is at the level of low SVC. Note 3: REASON FOR EXAMINATION: Dyspnea. Portable AP chest radiograph was compared to a prior study obtained on ___. Current study demonstrates interval progression of pulmonary edema, interstitial on the top of previously described consolidations and pleural effusion. Note 4: AP CHEST 8:39 A.M. ON ___ HISTORY: An ___ man re-admitted with shortness of breath after MVR. Evaluate for effusion. IMPRESSION: AP chest compared to ___: Moderate to large right pleural effusion not changed appreciably since ___. Left lower lobe remains collapsed accompanied by small to moderate pleural effusion on that side. No pneumothorax. Heart is very large but stable post-operatively and there is no pulmonary edema. Note 5: CHEST RADIOGRAPH INDICATION: Chest tube placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, a right-sided chest tube has been placed. The tube is in correct position. The extent of right pleural effusion has slightly decreased. There is a small extrathoracic air collection at the site of tube insertion. No evidence of right-sided pneumothorax. Unchanged appearance of the cardiac silhouette and of the left lung base, with an extensive left basal atelectasis. No newly appeared focal parenchymal opacity suggesting pneumonia. Note 6: CHEST CT ON ___ HISTORY: Pleural effusion and bacteremia. TECHNIQUE: Multidetector helical scanning of the chest was performed without intravenous contrast agent reconstructed as contiguous 5- and 1.25-mm thick axial and 5-mm thick coronal and parasagittal images, read in conjunction with chest radiographs ___ through ___: FINDINGS: Moderate size nonhemorrhagic bilateral pleural effusion layers posteriorly. A moderate-sized pericardial effusion greatest in the region normally occupied by the left atrial appendage is separated from cardiac chambers by preserved epicardial fat. The attenuation of the pericardial and pleural effusions though below that of frank hemorrhage is above that of simple transudate, which is what one would expect following surgery where fluid has been present for several days and may have small components of blood. There is no radiographic evidence of tamponade physiology. Atelectasis is present in all of the basal segments of the left lower lobe and the posterior basal and some of the lateral basal on the right. Central bronchi are not obstructed. Right apical and anterior pneumothorax is very small. Right pleural tube running up the major fissure to the upper posterior hemithorax, is filled with material. Whether it is occluded can be judged clinically, but conceivably it is isolated from the right pleural effusion. There is no pneumonia. Sternal fragments are closely applied in the body of the sternum, more separated in the manubrium, but the postoperative appearance at all levels is unremarkable and there is no associated fluid collection. Heart is severely enlarged, particularly the atria. Mitral annulus is heavily calcified. IMPRESSION: 1. Moderate bilateral pleural effusions are dependent, with no good evidence for loculation or acute hemorrhage. Differentiation of empyema from persistent, noninfectious, postoperative effusion is not possible radiographically, but should be feasible with image-directed thoracentesis. Pericardial effusion is moderate, with no evidence of tamponade physiology. Right pleural tube is fissural and may be isolated from the right pleural effusion and small right pneumothorax. 2. Bibasilar atelectasis attributable to the pleural effusion. No evidence of pneumonia. 3. Moderate-to-severe cardiomegaly predominantly atrial. MVR within heavily calcified mitral annulus. Note 7: PORTABLE CHEST, ___ COMPARISON: Radiograph of ___. FINDINGS: Cardiac silhouette remains enlarged. Right-sided chest tube remains in place with slight decrease in size of small right pleural effusion with associated improving atelectasis at the right base. No definite pneumothorax. Moderate left pleural effusion and adjacent left lower lobe opacity appears similar to the prior study. Note 8: CHEST X-RAY HISTORY: Bilateral pleural effusions, assess change. One view. Comparison with the previous study done ___. There is continued evidence of small pleural effusions, greater on the left, probably unchanged. Increased density in the retrocardiac area consistent with atelectasis or consolidation persists. The patient is rotated to the left as before. Mediastinal structures are unchanged. IMPRESSION: No significant interval change. Note 9: CHEST HISTORY: Chest tube removal, rule out pneumothorax. One view. Comparison with the previous study done earlier the same day. A right chest tube has been removed. No pneumothorax is identified. The right costophrenic sulcus is blunted consistent with small pleural effusion as before. There is a larger pleural effusion on the left. There is increased density in the underlying left lower lobe consistent with atelectasis and/or consolidation. The patient is rotated to the left. The cardiac silhouette is prominent. A prosthetic mitral valve is in place. Mediastinal structures are stable. IMPRESSION: No significant change post right chest tube removal. Note 10: CLINICAL HISTORY: Status post mitral valve replacement, chest tube removed. CHEST: The right lung shows no pneumothorax. The right costophrenic angle is sharp. Left pleural effusion is present, best seen on the lateral film. Calcification of mitral annulus is present. Left lower lobe atelectasis is probably also present. Marked scoliosis of the thoracic spine is again noted. IMPRESSION: No pneumothorax on right side. Left effusion and probable atelectasis persists. Note 11: PICC LINE PLACEMENT INDICATION: IV access needed for antibiotics. The procedure was explained to the patient. A timeout was performed. RADIOLOGIST: Drs. ___ performed the procedure. TECHNIQUE: Using sterile technique and local anesthesia, the right basilic vein was punctured under direct ultrasound guidance using a micropuncture set. Hard copies of ultrasound images were obtained before and immediately after establishing intravenous access are on file. A peel-away sheath was then placed over a guide wire, and a 4 ___ single lumen PICC line measuring 39 cm in length was then placed through the peel-away sheath with its tip positioned in the SVC under fluoroscopic guidance. The position of the catheter was confirmed by a fluoroscopic spot film of the chest. The peel-away sheath and guide wire were then removed. The catheter was secured to the skin, flushed, and a sterile dressing applied. The patient tolerated the procedure well. There were no immediate complications. IMPRESSION: Uncomplicated ultrasound and fluoroscopically guided 4 ___ single lumen PICC line placement via the right basilic venous approach. Final internal length is 39 cm, with the tip positioned in SVC. The line is ready to use. Note 12: REASON FOR EXAMINATION: Evaluation of the patient after mitral valve repair. Portable AP chest radiograph was reviewed in comparison to ___. Cardiomediastinal silhouette is unchanged. There is excentric mitral valve calcification. The left pleural effusion is redemonstrated. The left retrocardiac atelectasis is seen. Note is made that the right costophrenic angle was not included in the field of view. Overall no substantial change since the prior study has been demonstrated. Note 13: CHEST RADIOGRAPH INDICATION: Chronic heart failure, evaluation of the cardiac silhouette. COMPARISON: ___, 7:47 a.m. FINDINGS: As compared to the previous radiograph, there is unchanged evidence of moderate cardiomegaly with extensive retrocardiac atelectasis. Unchanged presence of bilateral pleural effusions. Marked signs of parenchymal overinflation, but no interval appearance of new focal parenchymal opacities. Unchanged course and position of the right-sided PICC line. Note 14: CHEST RADIOGRAPH INDICATION: Respiratory distress, line placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the patient has been intubated. The tip of the endotracheal tube projects 3 cm above the carina. The course of the nasogastric tube is unremarkable. The right PICC line is in unchanged position. Status post insertion of a right subclavian vein introduction sheath. No pneumothorax. Unchanged small left pleural effusion, unchanged retrocardiac and right basal atelectasis. Note 15: CHEST RADIOGRAPH INDICATION: Status post Swan-Ganz placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the Swan-Ganz catheter has been introduced over the venous introduction sheath positioned in the subclavian vein. The catheter is coiled at the transition zone between left atrium and ventricle. The referring physicians were notified about the need for catheter repositioning. No complications, notably no pneumothorax. Otherwise, the radiograph is unchanged. Note 16: CHEST RADIOGRAPH INDICATION: Swan-Ganz placement. Evaluation for position. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the Swan-Ganz catheter is still coiled in the right atrium. Repositioning of the device is required. There is no evidence of complication. The other monitoring and support devices are in unchanged position. At the time of dictation, 3:49 p.m., ___, the referring physician, ___ was paged for notification. Subsequently, the findings were discussed over the telephone. Note 17: INDICATION: ___ man status post mitral valve replacement. Evaluate for loculated effusion or infiltrate. COMPARISON: CT of the chest from ___. TECHNIQUE: MDCT images were acquired without contrast through the chest. Lung reconstruction algorithms, thin sections and multiplanar reformations were obtained and reviewed. CT OF THE CHEST WITHOUT IV CONTRAST: The thyroid gland is unremarkable. There is a small amount of air within the right internal jugular vein. A right PICC and IJ catheter terminate in the SVC appropriately. Air is also noted within the right ventricle. The heart shows a small pericardial effusion (small compared to the prior exam) and diffuse mitral annular calcifications. Radiopaque markers indicate a mitral valve replacement. The patient is status post sternotomy and sternal wires are intact. The airways are patent down to the subsegmental level except the right and left lower lobes where dense consolidative atelectasis is noted. There is left greater than right pleural effusions with associated compressive atelectasis. The left sided effusion has increased compared to prior. Right sided effusion has decreased slightly. Interval removal of right sided chest tube. The prior noted pneumothorax has now resolved. There is a right major fissural 13 mm loculated fluid collection (2:23). No anterior mediastinal fluid collection is noted. There is mild associated ground glass opacitied with smooth interlobular septal thickening adjacents to the atelectatic area. Although this examination was not intended for subdiaphragmatic evaluation, the partially imaged abdomen shows an unremarkable liver, spleen, both adrenals, and both kidneys. There is sludge within the gallbladder with a large stone in the gallbladder neck. No evidence of cholecystitis is noted. An NG tube terminates in the stomach appropriately. An ET tube terminates in the mid thoracic trachea appropriately as well. There is mild levocurvature of the thoracic spine. A 4 x 5.7 cm lobulated hyperdense lesion is noted along the left lower back, which is unchanged compared to the prior examination and may represent a larg sebaceous cyst. OSSEOUS STRUCTURES: The visible osseous structures show levocurvature of the thoracic spine and sternotomy wires, which are intact, but no fractures, suspicious lytic or blastic lesions are noted. IMPRESSION: 1. Left greater than right pleural effusions, slightly decreased on the right and increased on the left, with associated compressive atelectasis and likely mild pulmonary edema. 2. 4 x 5.7 lobulated hyperdense lesion in the subcutaneous tissues of the left lower back may represent a sebaceous cyst, but other entities are not excluded and this is not fully characterized. An ultrasound may be obtained for further evaluation if clinically indicated. Note 18: INDICATION: Left effusion, status post thoracentesis. COMPARISON: Multiple prior chest radiographs, most recently chest CT ___. FINDINGS: There is improved aeration of the left upper lung field. This may relate to a more upright position since prior film or due to decreased layering fluid, status post thoracentesis. There is a small left pleural effusion. There is unchanged appearance to the right lung fields. The tip of the endotracheal tube has been pulled back, now 6 cm from the carina. There is a right subclavian line with the tip terminating at the low SVC. There is an orogastric tube with the tip not visible on this film. There are sternotomy wires and mitral valve calcifications. There is stable cardiomegaly. IMPRESSION: Interval improved aeration of the upper lung fields concordant with recent thoracentesis. Small persistent left pleural effusion. Note 19: HISTORY: Elevated total bilirubin with normal LFTs. Evaluate for findings of cholestasis. COMPARISON: ___ and ___ chest CT. LIMITED ABDOMINAL ULTRASOUND: The liver parenchyma is homogeneous with a single small 1.4 x 0.9 x 1.3 cm cyst noted within the right lobe. Portal vein is patent with normal hepatopetal flow. No biliary ductal dilatation is present with the common duct measuring 2 mm. The gallbladder is moderately dilated, sludge-filled and contains at least two rim calcified stones. Marked wall edema is present but without any significant hypervascularity. The right kidney measures 11.8 cm and left kidney measures 11 cm. No hydronephrosis or large calculi are present. A simple cortical lower pole right renal cyst is present measuring 2.4 x 2.6 x 2.6 cm. IMPRESSION: Sludge and stone-filled gallbladder with wall edema. This may reflect underlying cholestasis with wall edema related to known CHF, however, acute cholecystitis can have a similar appearance. If there is a high clinical concern for cholecystitis then suggest correlation with a HIDA scan. These findings were discussed with caring provider ___ on date of exam via phone by Dr. ___ at 12:08 p.m. Note 20: INDICATION: Status post Swan-Ganz catheter placement and Dobbhoff tube. COMPARISON: Multiple prior chest radiographs, most recently earlier in the same day at 9:00 a.m. FINDINGS: Single portable AP chest radiograph was obtained. There is a Dobbhoff tube which coils in the mid esophagus. A Swan-Ganz catheter has been inserted with its tip at the right main pulmonary artery. There is stable position of an endotracheal tube and an orogastric tube and a right PICC line in the mid SVC. Stable appearance of the small left pleural effusion and stable cardiomegaly. IMPRESSION: Dobbhoff tube which is coiled in the mid thorax, appropriate position of Swan-Ganz catheter. Findings were communicated with Dr. ___ at 1:30 p.m. on ___. Note 21: HISTORY: Dobbhoff placement. FINDINGS: In comparison with study of ___, the Dobbhoff tube now extends to distal stomach. Otherwise, little change. Note 22: HISTORY: Dobbhoff placement. FINDINGS: In comparison with the study of ___, the tip of the Dobbhoff tube is in the distal stomach. Little overall change in the appearance of the heart and lungs with extensive opacification in the retrocardiac region. Note 23: HISTORY: Dobbhoff repositioned. FINDINGS: In comparison with the earlier study of this date, the tip of the Dobbhoff tube has been pulled back to the mid body of the stomach. Remainder of the study is unchanged. Note 24: CHEST HISTORY: Shortness of breath. Bacteremia and endocarditis. Single AP view of the chest performed at 8:35 a.m. is compared to prior study performed ___ at 1815. Tip of the feeding tube remains in the stomach. Tip of the right-sided PICC line is at the cavoatrial junction. There is blunting of the right costophrenic angle essentially unchanged. Increased opacity at the right lung base is most consistent with atelectasis. The heart is enlarged. There is increased opacity in the left costophrenic angle likely secondary to increasing pleural effusion. There is increased opacity in the retrocardiac area which is unchanged. IMPRESSION: Tubes and lines in adequate position. Compared to the prior study there is likely increased opacity in the left costophrenic angle consistent with increasing pleural effusion. Dense retrocardiac opacity persists. New right lower lobe opacity most consistent with atelectasis. Note 25: INDICATION: An ___ male with bacteremia and endocarditis, as well as left upper extremity edema. Evaluate for DVT. COMPARISON: None. FINDINGS: Grayscale and color sonographic imaging of the left internal jugular, subclavian, axillary, brachial, basilic, and cephalic veins was performed. The veins demonstrate normal compressibility, flow, and augmentation. There is no echogenic intraluminal thrombus seen. The contralateral subclavian vein was interrogated for comparison purposes, demonstrating symmetric respiratory phasicity. IMPRESSION: No left upper extremity DVT. Note 26: AP CHEST, 9:56 A.M., ___ HISTORY: An ___ man status post MVR. Dobbhoff tube placement. IMPRESSION: AP chest compared to ___ through ___: Feeding tube ends in the region of the pylorus. Chest is otherwise essentially unchanged over several days, including the chronically enlarged heart, collapsed left lower lobe, small left pleural effusion, borderline edema in the right upper lobe, and persistent right lower lobe consolidation concerning for pneumonia. Right PIC line ends in the low SVC. No pneumothorax. Note 27: CHEST RADIOGRAPH INDICATION: Assessment of Swan-Ganz catheter. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the patient was intubated. The tip of the endotracheal tube projects 4.4 cm above the carina. The patient also received a left subclavian venous introduction sheath. There is no safe evidence of pneumothorax. The nasogastric tube has been exchanged, the tip of the current tube projects over the middle parts of the stomach. Unchanged course and position of the right PICC line. Unchanged size of the cardiac silhouette. Unchanged small left pleural effusion and retrocardiac atelectasis. Improved ventilation of the right lung with minimal right basal pleural effusion. No newly occurred focal parenchymal opacities. Note 28: ULTRASOUND ABDOMEN INDICATION: Post-mitral valve replacement, right upper quadrant pain and hematocrit drop. COMPARISON: Comparison is made with prior ultrasound performed ___. FINDINGS: Technically diffivult examination due to patient body habitus. There is a large shadowing calculus identified within the neck of the gallbladder. There is extensive gallbladder wall thickening and pericholecystic fluid with gallbladder distension and sludge in keeping with acute cholecystitis. Common bile duct measures 7 mm. No intrahepatic biliary dilatation is identified. There is a large collection identified in the retroperitoneum inferior to the right lobe of the liver and posterior to the right kidney it measures 7.8 x 18 cm in length and contains internal echogenicity and septations consistent with a retroperitoneal hematoma on the right side. A 1.4-cm simple cyst is identified mainly in the right lobe of the liver. IMPRESSION: 1. Acute cholecystitis. 2. Large retroperitoneal hematoma (right). Case discussed with ___ in person. Note 29: INDICATION: ___ male status post mitral valve replacement with question of mesenteric ischemia. COMPARISON: CT of the chest without IV contrast from ___. TECHNIQUE: 64-row MDCT obtained of the abdomen and pelvis with images from the lung bases through to the proximal femora with only oral contrast. FINDINGS: There are bilateral pleural effusions in the lungs, left greater than right, with associated compressive atelectasis seen in the left lung. There is no pulmonary nodule. In the aerated lung, there is no focal consolidation to suggest pneumonia. There is significant metallic artifact seen at the location of the known replaced mitral valve. Calcifications of the LAD are noted. There is a nasoenteric tube seen with the tip ending in the stomach. Again noted is a 1.3 x 1.2 cm hypodensity in segment V of the liver that is unchanged from prior non-contrast CT. Otherwise, evaluation of the intra-abdominal solid organs is limited without IV contrast. There is ascites seen around the liver. The unenhanced kidneys, pancreas, and adrenals and spleen are unremarkable. There is a large right retroperitoneal hematoma tracking along the psoas measuring 11 cm AP x 11 cm craniocaudal x 15 cm in transverse direction with a hematocrit level suggesting acute on subacute evolving hematoma. There is no evidence of free air within the abdomen. There are numerous gallstones within the gallbladder as well as a distended gallbladder with gallbladder wall edema. There is no evidence of intra- abdominal wall distention or bowel wall thickening. Again seen is a 2.8-cm hypodensity within the right kidney that is appreciated on the earlier chest CT and is unchanged in size. Note is made of atherosclerotic calcification seen in the infrarenal aorta as well as the common iliacs. CT PELVIS WITH ORAL CONTRAST: Patient has a catheter within the right femoral artery with no associated hematoma. There also is a catheter seen within the left femoral vein with the tip ending proximally at the bifurcation of the external and internal iliac. There is an intrapelvic fluid collection measuring roughly 7.6 x 9.3 cm (300B, 42) with what appears to be a layering hematocrit effect. The bladder is decompressed with a Foley catheter in place. The prostate is unremarkable. There is diffuse stranding of the subcutaneous tissues consistent with anasarca. OSSEOUS STRUCTURES: There is a levocurvature of the thoracic spine. There are no fractures or suspicious lytic or blastic lesions to suggest metastatic disease. IMPRESSION: 1. Large retroperitoneal hematoma with layering hematocrit suggesting acute on chronic component to this evolving hematoma. 2. Free fluid within the rectovesicular space with question of layering hematocrit effect suggestive of a possible bowel perforation. 3. Distended gallbladder with gallbladder wall edema concerning for acute cholecystitis in the appropriate clinical setting 4. No definitive bowel wall distention or bowel wall thickening to suggest ischemia; however, as mentioned, there is an intrapelvic fluid collection. Note 30: PROCEDURE: Ultrasound-guided percutaneous cholecystostomy. INDICATION: Acute cholecystitis, sepsis, and hypotension. COMPARISON: Comparison has been made with prior CT, ___ and ultrasound, ___. PROCEDURE: The risks, benefits, and alternatives to the procedure were explained to the patient's wife, and verbal consent was obtained over the phone. The procedure was performed in the cardiovascular ICU with portable ultrasound. Under ultrasound guidance, an entrance site was selected, and the skin was prepped and draped in the usual sterile fashion. A preprocedure timeout was performed using a single patient identifier. 1% buffered lidocaine was instilled for local anesthesia. US of the gallbladder demonstrated sludge and gallstones with gallbladder wall thickening and pericholecystic fluid.There was also small trace of ___ ascites. An 8fr catheter was inserted under ultrasound guidance via a subcostal approach. 65 cc of dark reddish bile was aspirated, and catheter was left on free drainage at the end of the procedure. Sample of bile was sent to microbiology for culture and sensitivity. The patient was ventilated and sedated in the ICU. Patient tolerated the procedure well, and there were no immediate complications. Dr. ___ attending radiologist, and Dr. ___, the fellow, were present throughout the procedure. Post-procedure instructions were written in the ___ medical record. IMPRESSION: Technically successful ultrasound-guided percutaneous cholecystostomy. Note 31: AP CHEST, 11:46 A.M., ___ HISTORY: MVR. Line placement. IMPRESSION: AP chest compared to ___: Tip of the new left subclavian line ends alongside the right PICC line at the junction of the brachiocephalic veins. Mild interstitial edema is new accompanied by increase in heart size, still at the upper limits of normal and new small left pleural effusion. There is no pneumothorax. Left lower lobe is still airless. ET tube is in standard placement. No pneumothorax. Note 32: AP CHEST, 1:10 P.M. ON ___ HISTORY: ___ man after mitral valve repair. IMPRESSION: AP chest compared to ___ through ___: Mild pulmonary edema present on ___ has improved. Left lower lobe collapse and moderate left pleural effusion has not. There is no pneumothorax. Mild-to-moderate cardiomegaly is stable, and mediastinum is not widened. ET tube is in standard position, nasogastric tube ends in the distal stomach, and a left subclavian line ends alongside a right PIC line in the upper SVC. No pneumothorax. Note 33: AP CHEST 6 P.M. ON ___ HISTORY: Renal failure. IMPRESSION: Left subclavian line ends at the junction of brachiocephalic veins, dual-channel right internal jugular line ends in the low SVC. No pneumothorax, new mediastinal widening or right pleural effusion. Pulmonary vascular engorgement has improved, previous mild interstitial edema has resolved, left lower lobe atelectasis has improved and previous moderate left pleural effusion has decreased. Heart size top normal. ET tube in standard placement. Nasogastric tube ends in the distal stomach. Right upper quadrant drainage catheter noted but cannot be localized on this single frontal view. Note 34: HISTORY: Postoperative complications with new hemoptysis. FINDINGS: In comparison with the study of ___, there is continued vascular engorgement with some left basilar atelectasis and pleural effusion in a patient with dense calcification in the mitral region. Tracheostomy tube is in good position as are the central catheters. Note 35: LIVER ULTRASOUND INDICATION: History of prior percutaneous cholecystostomy with rising bilirubin. COMPARISON: Ultrasound percutaneous cholecystostomy ___, CT abdomen and pelvis ___, ultrasound abdomen ___. FINDINGS: The cholecystostomy tube is seen within the gallbladder lumen. The gallbladder is decompressed with extensive gallbladder wall thickening noted. A small amount of pericholecystic fluid is also noted. Multiple shadowing gallstones and sludge noted within the gallbladder. Tiny trace of perihepatic free fluid is noted. There is no evidence of intrahepatic biliary dilatation. The common bile duct measures 5 mm. There is normal liver echotexture. There is a 1.0 x 0.9 x 1.3 cm hypoechoic lesion identified within the right lobe of the liver, likely simple hepatic cysts. Pancreas visualized in the midline though body and tail are not visualized in their entirety. Spleen is normal in caliber. Note is made of atelectasis / consolidation of the left lower lobe with associated small left effusion. IMPRESSION: 1. Decompressed gallbladder with cholecystostomy tube noted within the gallbladder wall lumen. Sludge and stones noted within the lumen.Trace pericholecystic fluid noted. 2. No intrahepatic or extrahepatic biliary dilatation. 3. Simple hepatic cyst. Note 36: INDICATION: ___ man with history of renal failure, for dialysis. PHYSICIAN: Dr. ___, the attending radiologist, performed the procedure. PROCEDURES: 1. Initial fluoroscopic spot image. 2. Placement of a 23-cm tip-to-cuff tunneled dialysis catheter via the right internal jugular vein. 3. Post-placement fluoroscopic spot image. MEDICATIONS: The patient received moderate sedation with two divided doses of 15 mcg of fentanyl IV and two divided doses of 1 mg IV Versed throughout the intraservice time of 20 minutes, during which continuous hemodynamic monitoring was performed by a trained radiology nurse. Additionally, 1% local lidocaine was administered. PROCEDURE: Prior to initiation of procedure, written informed consent was obtained and a preprocedure timeout was performed. The right upper neck was prepped and draped in a sterile manner. Under ultrasound guidance, micropuncture access was obtained into the right internal jugular vein. Pre- and post-access hard copy ultrasound images were obtained are on file. A site along the anterior chest wall was then selected, and after local lidocaine administration was performed, a tunneling device was used to tunnel the catheter to the venous access site. Next, sequential fascial dilatation of the venotomy was performed over the existing guidewire, and a peel-away sheath was placed. The catheter was advanced through the peel-away sheath, such that the tip was positioned in the right atrium. The venous access site was sutured with ___ Vicryl suture. Catheter was secured with ___ silk suture. The catheter aspirated and flushed well and was in good location on the final fluoroscopic spot image. The patient tolerated the procedure well. IMPRESSION: Successful placement of a tunneled hemodialysis catheter (23-cm tip-to-cuff) via the right internal jugular vein with tip in the right atrium. The catheter is ready to use. Note 37: INDICATION: ___ male with bacteremia and poor aeration on the left side. COMPARISON: ___. TECHNIQUE: Single AP radiograph of the chest was obtained with the patient in the upright position. FINDINGS: There is complete opacification of the left lung with shift of mediastinal structures towards the left, consistent with collapse. Evaluation of the left pleural effusion cannot be performed in this setting. The right lung demonstrates pulmonary vascular congestion and a small effusion. Extensive calcification of the mitral annulus and mitral valve repair are visualized. The tracheostomy tube is visualized in similar position compared to prior. There is a right-sided central catheter with tip in the right atrium. Sternal wires are noted. There has been interval removal of the left subclavian catheter. A right-sided PICC catheter is seen with tip at the confluence of the brachiocephalic veins. Severe scoliosis is seen. IMPRESSION: Left lung collapse. These findings were discussed with ___, NP by Dr. ___ by telephone at 10:40 a.m. on ___. Note 38: INDICATION: ___ male with bacteremia and left lung collapse, now status post bronchoscopy. ___ at 08:00 a.m. TECHNIQUE: Single AP radiograph of the chest was obtained. FINDINGS: The lung apices are not included in this view. Compared to most recent prior, there has been partial re-expansion of the left upper lung. There is persistent opacification of the left lower hemithorax. There is a left pleural effusion which is incompletely evaluated in the setting of partial left lung collapse. The right lung is unchanged. Tracheostomy tube, central line with tip in the right atrium, mitral annular calcification and sternal wires are again seen. IMPRESSION: Partial reexpansion of the left upper lung with persistent collapse of the left lower lobe. These findings were discussed with ___, NP by Dr. ___ by telephone at approximately 10:45 a.m. on ___. | false | 0 |
| 1 | Note 1: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with chest tubes// Assess placement Assess placement IMPRESSION: Comparison to ___. The bilateral chest tubes are in stable position. Stable extent and severity of the pre-existing bilateral parenchymal opacities, right more than left, no evidence of pneumothorax. Stable mild cardiomegaly. Note 2: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with PTX s/p bilateral chest tubes// Assess for placement, remaining air Assess for placement, remaining air IMPRESSION: Comparison to ___. The patient has been extubated. The feeding tube and the bilateral chest tubes are in stable position. A right pneumothorax is approximately 1 cm in diameter. On the left, no pneumothorax is currently visualized. Slightly increasing extent of a retrocardiac atelectasis. Otherwise unchanged appearance of the lung parenchyma. Stable mild cardiomegaly. No evidence of tension. Note 3: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with bilateral PTX and chest tube, removed left tube, right with persistent PTX.// Assess for change in right sided PTX and any new PTX after left chest tube removal? Assess for change in right sided PTX and any new PTX after left chest tube removal? IMPRESSION: After removal of the left chest tube, a 1 cm left apical pneumothorax has reoccurred. There is no evidence of tension. On the right, despite the present chest tube, a 1 cm right apical pneumothorax persists. Again, there is no evidence of tension. Stable parenchymal opacities. Stable moderate cardiomegaly. Note 4: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with bilateral PTX, right chest tube in place, left removed// stability of bilateral PTX IMPRESSION: In comparison with the study of ___, after removal of the left chest tube any residual pneumothorax is quite small. Right chest tube remains in place with little change change in the small apical pneumothorax. The left hemidiaphragmatic contour is more sharply seen, suggesting decreasing volume loss and pleural fluid, with similar changes on the right. A similar appearance could merely reflect a more upright position of the patient. Small amount of subcutaneous gas is again seen along the right lateral chest wall. Note 5: EXAMINATION: Chest radiograph, portable AP upright. INDICATION: Query pneumothorax. Chest tube in place. COMPARISON: Prior study from earlier on the same day. FINDINGS: Right-sided chest tube and PICC lines appear unchanged in position. Previously, pneumothorax was very small, measuring only up to 5 mm in with at the lung apex but now up to 23 mm. There is probably a small pleural fluid component, and increased opacification and volume loss at the right lung base suggest atelectasis. Streaky basilar opacities on the left also suggests minor atelectasis with a trace pleural effusion. IMPRESSION: Increasing right pneumothorax, now mild to moderate. NOTIFICATION: The findings were discussed with Dr. ___ , M.D. by ___, M.D. on the telephone on ___ at 5:53 pm, 2 minutes after discovery of the findings. Note 6: EXAMINATION: Chest radiograph, portable AP upright INDICATION: Follow-up pneumothorax. COMPARISON: Earlier on the same evening. FINDINGS: There is a tiny right apical pneumothorax, considerably decreased and largely resolve. Chest tube and PICC line appear unchanged. Right basilar atelectasis has largely resolved. There is similar minor left basilar atelectasis. Trace pleural fluid is probably unchanged bilaterally. Trace subcutaneous emphysema appears unchanged bilaterally. IMPRESSION: Marked decrease in right pneumothorax. Note 7: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with right chest tube in place, has continued to be on low wall suction// ?interval change IMPRESSION: In comparison with the study of the ___, with the right chest tube on suction, there is little change in the small apical pneumothorax. A the minimal subcutaneous gas remains along the right lateral chest wall. The opacification at the bases most likely represents residual atelectatic changes. Note 8: EXAMINATION: Chest radiograph INDICATION: ___ year old man with bilateral PTX// assess for interval change. PLEASE OBTAIN AT 1230 TECHNIQUE: Chest PA and lateral COMPARISON: Multiple prior chest radiographs, most recently ordered today. FINDINGS: Unchanged position of right-sided chest tube and PICC line. Minimal change in bilateral pneumothorax. No pleural effusions. Mild bibasilar atelectasis. Heart size remains normal with unremarkable cardiomediastinal silhouette. IMPRESSION: No significant interval change of bilateral small apical pneumothorax. Note 9: INDICATION: ___ year old man with acute hypoxic respiratory failure/PNA c/b bilateral pneumothoraces s/p bilateral chest tubes. Left removed ___, right ___// evaluate for stability/improvement in bilateral pneumothoraces TECHNIQUE: Chest PA and lateral COMPARISON: ___ IMPRESSION: The small right-sided pneumothorax is unchanged. Right-sided chest tube has been removed in the interim. Right-sided PICC line projects to the distal SVC. There is minimal subsegmental atelectasis in both lung bases. Cardiomediastinal silhouette is stable. Trace left pneumothorax has slightly improved Note 10: INDICATION: ___ year old man with hypoxic resp failure c/b bilateral PTX now s/p CT removal// pls evaluate for interval changes/improvement in PTX TECHNIQUE: Chest PA and lateral COMPARISON: Chest x-ray ___ FINDINGS: Redemonstration of a right-sided PICC, which terminates in the mid SVC. Interval reduction in bilateral chest wall subcutaneous emphysema. Miniscule residual of pleural air, if any, right apex and left base. Tiny left pleural effusion has almost resolved. Small area of atelectasis or persistent consolidation in left lung laterally, developed between ___ and ___. Basal linear atelectasis is improving. Lungs otherwise essentially clear. Cardiomediastinal silhouette is normal. Bilateral hila are unremarkable. IMPRESSION: 1. Miniscule if any pneumothorax, right apex, left base. 2. Small area of atelectasis or persistent consolidation in the left lower lung. 3. Interval near resolution tiny left pleural effusion and improved bibasilar atelectasis. Note 11: EXAMINATION: CHEST (PORTABLE AP) INDICATION: History: ___ with Please confirm chest tube placement// Please confirm chest tube placement TECHNIQUE: Single frontal view of the chest COMPARISON: None FINDINGS: Endotracheal tube terminates 6.2 cm above the carina. Enteric tube courses below the diaphragm, out of the field of view. There are bilateral chest tubes. There are extensive bilateral pulmonary opacities. Blunting of the costophrenic angles may be due to pleural effusions. Cardiac and mediastinal silhouettes are grossly unremarkable. IMPRESSION: Endotracheal tube terminates 6.2 cm above the carina. Enteric tube courses below the diaphragm, out of the field of view. Extensive bilateral nodular pulmonary opacities with no prior available for comparison. Findings may be due to extensive multifocal pneumonia, ARDS, underlying malignancy not excluded. Blunting of the costophrenic angles could be due to pleural effusions and/or atelectasis. Note 12: EXAMINATION: CT torso with intravenous contrast. INDICATION: ___ with severe sepsis and ARDS// e/o cholangitis? intraabdominal processes TECHNIQUE: MDCT axial images were acquired through the chest, abdomen and pelvis following intravenous contrast administration with split bolus technique. Oral contrast was not administered.Coronal and sagittal reformations were performed and reviewed on PACS. DOSE: Total DLP (Body) = 609 mGy-cm. COMPARISON: None. FINDINGS: CHEST: Endotracheal tube appears to terminate 4.3 cm above the carina. The NG tube terminates in the stomach. Bilateral chest tubes terminate at the apex. Extensive bilateral pulmonary consolidations most notably in the lower lungs consistent with multifocal pneumonia. Thoracic aorta is normal in course and caliber. The main pulmonary artery and central branches are patent. No lymphadenopathy is seen. Heart is normal in size and shape. No pleural or pericardial effusion is seen. There is no residual pneumothorax. ABDOMEN: Anomalous anatomy in the upper abdomen likely reflect congenital absence of the right portal vein with resultant atrophy. Small bowel is seen clustered in the right subdiaphragmatic space. The liver measures 19 cm in length. No discrete focal liver lesion seen. Main portal vein is patent. The gallbladder contains a gallstone and is decompressed. There is no biliary ductal dilation. The spleen is mildly enlarged measuring 14 cm in length. The kidneys appear normal though both appear slightly displaced superiorly. Adrenal glands appear unremarkable. The abdominal aorta is normal in course and caliber. Anomalous vascular anatomy noted a common trunk from the abdominal aorta giving rise to the right renal artery, celiac trunk and SMA. The pancreas appears grossly unremarkable. The stomach is unremarkable. The duodenum follows an unusual course extending superiorly into the right upper abdomen. The small bowel is difficult to trace beyond the proximal duodenum in the absence of oral contrast. Malrotation is difficult to exclude. No lymphadenopathy. There is a small volume free fluid. No free air. PELVIS: No signs of ileus or obstruction. The appendix is normal. The colon is mostly decompressed. Small volume free fluid in the pelvis is noted. The urinary bladder is decompressed around a Foley catheter. There is no pelvic sidewall or inguinal adenopathy. A right common femoral venous catheter is in place. BONES: There is no acute fracture. No focal suspicious osseous abnormality. SOFT TISSUES: The abdominal and pelvic wall is within normal limits. IMPRESSION: 1. Extensive bilateral pulmonary airspace opacities most confluent in the lower lobes concerning for multifocal pneumonia. 2. Chest tubes in place without residual pneumothorax. 3. ET and NG tubes positioned appropriately. 4. Small volume ascites with enlarged liver measuring 18.5 cm. Splenomegaly also noted. Clinical correlation advised. 5. Anomalous anatomy in the abdomen likely reflecting congenital absence of the right portal vein and resultant right hepatic atrophy. 6. Anomalous branching anatomy from the abdominal aorta as described. 7. Small-bowel clustered in the right subdiaphragmatic space with possible small bowel malrotation. Note 13: EXAMINATION: DX CHEST PORT LINE/TUBE PLCMT 1 EXAM INDICATION: ___ year old man with ET tube, just advanced// check placement check placement IMPRESSION: Bilateral chest tube is present. NG tube tip is in the stomach. Multifocal extensive consolidations are similar to previous examination. Minimal right apical pneumothorax cannot be excluded. Note 14: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with resp failure on vent// Eval for changes, ETT placement Eval for changes, ETT placement IMPRESSION: Comparison to ___. The monitoring and support devices, including the bilateral chest tubes, are in stable correct position. The current radiograph shows no evidence for the presence of a pneumothorax. The parenchymal opacities, right more than left, are stable. No new parenchymal changes are noted. Stable moderate cardiomegaly. Note 15: INDICATION: ___ year old man with ARDS, PNA, pneumothraces s/p b/l chest tube. Increased tachacardia and hypoxemia// Pneumothorax, worsening effusions/infiltrates TECHNIQUE: AP portable chest radiograph COMPARISON: ___ FINDINGS: The tip of the endotracheal tube projects over the midthoracic trachea. A feeding tube extends to the stomach. Bilateral chest tubes are present. There is no discrete pneumothorax identified. Re-demonstrated are bilateral parenchymal opacities, increased in the right mid to lower lung and left base. The size of the cardiac silhouette is unchanged. IMPRESSION: Increasing bilateral parenchymal opacities. No discrete pneumothorax identified. Note 16: EXAMINATION: LIVER OR GALLBLADDER US (SINGLE ORGAN) INDICATION: ___ year old man with ards, persistent LFT derangements, hepatomegaly// Biliary dilation, assess for any obvious hepatic abnormalities TECHNIQUE: Grey scale and color Doppler ultrasound images of the abdomen were obtained. COMPARISON: CT abdomen and pelvis dated ___. FINDINGS: LIVER: The hepatic parenchyma appears within normal limits. The contour of the liver is smooth. There is no focal liver mass. The main portal vein is patent with hepatopetal flow. There is no ascites. BILE DUCTS: There is no intrahepatic biliary dilation. CHD: 3 mm GALLBLADDER: Cholelithiasis without gallbladder wall thickening. PANCREAS: The pancreas is not well visualized, largely obscured by overlying bowel gas. SPLEEN: Normal echogenicity. Spleen length: 9.7 cm KIDNEYS: Limited views of the kidneys show no hydronephrosis. RETROPERITONEUM: The visualized portions of aorta and IVC are within normal limits. IMPRESSION: 1. Normal appearing liver without biliary dilatation. 2. Cholelithiasis, but no evidence of acute cholecystitis. Note 17: EXAMINATION: CHEST PORT. LINE PLACEMENT INDICATION: ___ year old man with new R PICC// R DL Power PICC 37cm ___ ___ Contact name: ___: ___ R DL Power PICC 37cm ___ ___ IMPRESSION: ET tube tip is 5 cm above the carina. NG tube tip is in the stomach. Right PICC line tip is at the level of lower SVC. Right chest tube is in place. Left chest tube is in place. Unchanged are multifocal consolidations that are better appreciated on the recent chest CT from ___. | false | 0 |

This dataset is meant to train a large language model to following instructions to produce code from natural language. Each row in the dataset consists of an:

- instruction that describes a task
- input when additional context is required for the instruction, and
- the expected output.

> Instruction: "Based on the provided context, return true if the pation has ARDS, otherwise return false."
> Input: #MEDICAL NOTES FOR THE PATIENT#
> Output: True

We need to prepare the instruction and output for our dataset. We treat the output as strings.

```
DEMO_PROMPT = "Based on the following medical notes, please predict whether the patient described is likely to have Acute Respiratory Distress Syndrome (ARDS). Your prediction should be either 'true' if ARDS is likely, or 'false' if it is not likely."
df['instruction'] = DEMO_PROMPT
df = df.rename(columns={'text':'notes', 'label':'output'})
df = df.astype({'output':'str'})
df.head(1)

```

| index | notes | output | split | Instrunction |
|---|---|---| --- | --- |
| 0 | Note 1: ADDENDUM: In the impression, it states that the free fluid within the rectovesicular space with the layering hematocrit could be a possible bowel perforation. This is incorrect. This fluid collection in the pelvis represents an intraperitoneal extension of the right retroperitoneal hematoma and is not suggestive of bowel perforation. Note 2: REASON FOR EXAMINATION: Hypoxemia. Portable AP chest radiograph was reviewed in comparison to ___. There is interval progression of left lower lobe retrocardiac consolidation currently obscuring the entire left lower lobe and the hemidiaphragm. There is also progression of the right basal consolidation and new right pleural effusion demonstrated. Upper lungs are essentially unchanged. Replaced mitral valve projecting over the significantly calcified mitral annulus is redemonstrated. The right PICC line tip is at the level of low SVC. Note 3: REASON FOR EXAMINATION: Dyspnea. Portable AP chest radiograph was compared to a prior study obtained on ___. Current study demonstrates interval progression of pulmonary edema, interstitial on the top of previously described consolidations and pleural effusion. Note 4: AP CHEST 8:39 A.M. ON ___ HISTORY: An ___ man re-admitted with shortness of breath after MVR. Evaluate for effusion. IMPRESSION: AP chest compared to ___: Moderate to large right pleural effusion not changed appreciably since ___. Left lower lobe remains collapsed accompanied by small to moderate pleural effusion on that side. No pneumothorax. Heart is very large but stable post-operatively and there is no pulmonary edema. Note 5: CHEST RADIOGRAPH INDICATION: Chest tube placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, a right-sided chest tube has been placed. The tube is in correct position. The extent of right pleural effusion has slightly decreased. There is a small extrathoracic air collection at the site of tube insertion. No evidence of right-sided pneumothorax. Unchanged appearance of the cardiac silhouette and of the left lung base, with an extensive left basal atelectasis. No newly appeared focal parenchymal opacity suggesting pneumonia. Note 6: CHEST CT ON ___ HISTORY: Pleural effusion and bacteremia. TECHNIQUE: Multidetector helical scanning of the chest was performed without intravenous contrast agent reconstructed as contiguous 5- and 1.25-mm thick axial and 5-mm thick coronal and parasagittal images, read in conjunction with chest radiographs ___ through ___: FINDINGS: Moderate size nonhemorrhagic bilateral pleural effusion layers posteriorly. A moderate-sized pericardial effusion greatest in the region normally occupied by the left atrial appendage is separated from cardiac chambers by preserved epicardial fat. The attenuation of the pericardial and pleural effusions though below that of frank hemorrhage is above that of simple transudate, which is what one would expect following surgery where fluid has been present for several days and may have small components of blood. There is no radiographic evidence of tamponade physiology. Atelectasis is present in all of the basal segments of the left lower lobe and the posterior basal and some of the lateral basal on the right. Central bronchi are not obstructed. Right apical and anterior pneumothorax is very small. Right pleural tube running up the major fissure to the upper posterior hemithorax, is filled with material. Whether it is occluded can be judged clinically, but conceivably it is isolated from the right pleural effusion. There is no pneumonia. Sternal fragments are closely applied in the body of the sternum, more separated in the manubrium, but the postoperative appearance at all levels is unremarkable and there is no associated fluid collection. Heart is severely enlarged, particularly the atria. Mitral annulus is heavily calcified. IMPRESSION: 1. Moderate bilateral pleural effusions are dependent, with no good evidence for loculation or acute hemorrhage. Differentiation of empyema from persistent, noninfectious, postoperative effusion is not possible radiographically, but should be feasible with image-directed thoracentesis. Pericardial effusion is moderate, with no evidence of tamponade physiology. Right pleural tube is fissural and may be isolated from the right pleural effusion and small right pneumothorax. 2. Bibasilar atelectasis attributable to the pleural effusion. No evidence of pneumonia. 3. Moderate-to-severe cardiomegaly predominantly atrial. MVR within heavily calcified mitral annulus. Note 7: PORTABLE CHEST, ___ COMPARISON: Radiograph of ___. FINDINGS: Cardiac silhouette remains enlarged. Right-sided chest tube remains in place with slight decrease in size of small right pleural effusion with associated improving atelectasis at the right base. No definite pneumothorax. Moderate left pleural effusion and adjacent left lower lobe opacity appears similar to the prior study. Note 8: CHEST X-RAY HISTORY: Bilateral pleural effusions, assess change. One view. Comparison with the previous study done ___. There is continued evidence of small pleural effusions, greater on the left, probably unchanged. Increased density in the retrocardiac area consistent with atelectasis or consolidation persists. The patient is rotated to the left as before. Mediastinal structures are unchanged. IMPRESSION: No significant interval change. Note 9: CHEST HISTORY: Chest tube removal, rule out pneumothorax. One view. Comparison with the previous study done earlier the same day. A right chest tube has been removed. No pneumothorax is identified. The right costophrenic sulcus is blunted consistent with small pleural effusion as before. There is a larger pleural effusion on the left. There is increased density in the underlying left lower lobe consistent with atelectasis and/or consolidation. The patient is rotated to the left. The cardiac silhouette is prominent. A prosthetic mitral valve is in place. Mediastinal structures are stable. IMPRESSION: No significant change post right chest tube removal. Note 10: CLINICAL HISTORY: Status post mitral valve replacement, chest tube removed. CHEST: The right lung shows no pneumothorax. The right costophrenic angle is sharp. Left pleural effusion is present, best seen on the lateral film. Calcification of mitral annulus is present. Left lower lobe atelectasis is probably also present. Marked scoliosis of the thoracic spine is again noted. IMPRESSION: No pneumothorax on right side. Left effusion and probable atelectasis persists. Note 11: PICC LINE PLACEMENT INDICATION: IV access needed for antibiotics. The procedure was explained to the patient. A timeout was performed. RADIOLOGIST: Drs. ___ performed the procedure. TECHNIQUE: Using sterile technique and local anesthesia, the right basilic vein was punctured under direct ultrasound guidance using a micropuncture set. Hard copies of ultrasound images were obtained before and immediately after establishing intravenous access are on file. A peel-away sheath was then placed over a guide wire, and a 4 ___ single lumen PICC line measuring 39 cm in length was then placed through the peel-away sheath with its tip positioned in the SVC under fluoroscopic guidance. The position of the catheter was confirmed by a fluoroscopic spot film of the chest. The peel-away sheath and guide wire were then removed. The catheter was secured to the skin, flushed, and a sterile dressing applied. The patient tolerated the procedure well. There were no immediate complications. IMPRESSION: Uncomplicated ultrasound and fluoroscopically guided 4 ___ single lumen PICC line placement via the right basilic venous approach. Final internal length is 39 cm, with the tip positioned in SVC. The line is ready to use. Note 12: REASON FOR EXAMINATION: Evaluation of the patient after mitral valve repair. Portable AP chest radiograph was reviewed in comparison to ___. Cardiomediastinal silhouette is unchanged. There is excentric mitral valve calcification. The left pleural effusion is redemonstrated. The left retrocardiac atelectasis is seen. Note is made that the right costophrenic angle was not included in the field of view. Overall no substantial change since the prior study has been demonstrated. Note 13: CHEST RADIOGRAPH INDICATION: Chronic heart failure, evaluation of the cardiac silhouette. COMPARISON: ___, 7:47 a.m. FINDINGS: As compared to the previous radiograph, there is unchanged evidence of moderate cardiomegaly with extensive retrocardiac atelectasis. Unchanged presence of bilateral pleural effusions. Marked signs of parenchymal overinflation, but no interval appearance of new focal parenchymal opacities. Unchanged course and position of the right-sided PICC line. Note 14: CHEST RADIOGRAPH INDICATION: Respiratory distress, line placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the patient has been intubated. The tip of the endotracheal tube projects 3 cm above the carina. The course of the nasogastric tube is unremarkable. The right PICC line is in unchanged position. Status post insertion of a right subclavian vein introduction sheath. No pneumothorax. Unchanged small left pleural effusion, unchanged retrocardiac and right basal atelectasis. Note 15: CHEST RADIOGRAPH INDICATION: Status post Swan-Ganz placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the Swan-Ganz catheter has been introduced over the venous introduction sheath positioned in the subclavian vein. The catheter is coiled at the transition zone between left atrium and ventricle. The referring physicians were notified about the need for catheter repositioning. No complications, notably no pneumothorax. Otherwise, the radiograph is unchanged. Note 16: CHEST RADIOGRAPH INDICATION: Swan-Ganz placement. Evaluation for position. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the Swan-Ganz catheter is still coiled in the right atrium. Repositioning of the device is required. There is no evidence of complication. The other monitoring and support devices are in unchanged position. At the time of dictation, 3:49 p.m., ___, the referring physician, ___ was paged for notification. Subsequently, the findings were discussed over the telephone. Note 17: INDICATION: ___ man status post mitral valve replacement. Evaluate for loculated effusion or infiltrate. COMPARISON: CT of the chest from ___. TECHNIQUE: MDCT images were acquired without contrast through the chest. Lung reconstruction algorithms, thin sections and multiplanar reformations were obtained and reviewed. CT OF THE CHEST WITHOUT IV CONTRAST: The thyroid gland is unremarkable. There is a small amount of air within the right internal jugular vein. A right PICC and IJ catheter terminate in the SVC appropriately. Air is also noted within the right ventricle. The heart shows a small pericardial effusion (small compared to the prior exam) and diffuse mitral annular calcifications. Radiopaque markers indicate a mitral valve replacement. The patient is status post sternotomy and sternal wires are intact. The airways are patent down to the subsegmental level except the right and left lower lobes where dense consolidative atelectasis is noted. There is left greater than right pleural effusions with associated compressive atelectasis. The left sided effusion has increased compared to prior. Right sided effusion has decreased slightly. Interval removal of right sided chest tube. The prior noted pneumothorax has now resolved. There is a right major fissural 13 mm loculated fluid collection (2:23). No anterior mediastinal fluid collection is noted. There is mild associated ground glass opacitied with smooth interlobular septal thickening adjacents to the atelectatic area. Although this examination was not intended for subdiaphragmatic evaluation, the partially imaged abdomen shows an unremarkable liver, spleen, both adrenals, and both kidneys. There is sludge within the gallbladder with a large stone in the gallbladder neck. No evidence of cholecystitis is noted. An NG tube terminates in the stomach appropriately. An ET tube terminates in the mid thoracic trachea appropriately as well. There is mild levocurvature of the thoracic spine. A 4 x 5.7 cm lobulated hyperdense lesion is noted along the left lower back, which is unchanged compared to the prior examination and may represent a larg sebaceous cyst. OSSEOUS STRUCTURES: The visible osseous structures show levocurvature of the thoracic spine and sternotomy wires, which are intact, but no fractures, suspicious lytic or blastic lesions are noted. IMPRESSION: 1. Left greater than right pleural effusions, slightly decreased on the right and increased on the left, with associated compressive atelectasis and likely mild pulmonary edema. 2. 4 x 5.7 lobulated hyperdense lesion in the subcutaneous tissues of the left lower back may represent a sebaceous cyst, but other entities are not excluded and this is not fully characterized. An ultrasound may be obtained for further evaluation if clinically indicated. Note 18: INDICATION: Left effusion, status post thoracentesis. COMPARISON: Multiple prior chest radiographs, most recently chest CT ___. FINDINGS: There is improved aeration of the left upper lung field. This may relate to a more upright position since prior film or due to decreased layering fluid, status post thoracentesis. There is a small left pleural effusion. There is unchanged appearance to the right lung fields. The tip of the endotracheal tube has been pulled back, now 6 cm from the carina. There is a right subclavian line with the tip terminating at the low SVC. There is an orogastric tube with the tip not visible on this film. There are sternotomy wires and mitral valve calcifications. There is stable cardiomegaly. IMPRESSION: Interval improved aeration of the upper lung fields concordant with recent thoracentesis. Small persistent left pleural effusion. Note 19: HISTORY: Elevated total bilirubin with normal LFTs. Evaluate for findings of cholestasis. COMPARISON: ___ and ___ chest CT. LIMITED ABDOMINAL ULTRASOUND: The liver parenchyma is homogeneous with a single small 1.4 x 0.9 x 1.3 cm cyst noted within the right lobe. Portal vein is patent with normal hepatopetal flow. No biliary ductal dilatation is present with the common duct measuring 2 mm. The gallbladder is moderately dilated, sludge-filled and contains at least two rim calcified stones. Marked wall edema is present but without any significant hypervascularity. The right kidney measures 11.8 cm and left kidney measures 11 cm. No hydronephrosis or large calculi are present. A simple cortical lower pole right renal cyst is present measuring 2.4 x 2.6 x 2.6 cm. IMPRESSION: Sludge and stone-filled gallbladder with wall edema. This may reflect underlying cholestasis with wall edema related to known CHF, however, acute cholecystitis can have a similar appearance. If there is a high clinical concern for cholecystitis then suggest correlation with a HIDA scan. These findings were discussed with caring provider ___ on date of exam via phone by Dr. ___ at 12:08 p.m. Note 20: INDICATION: Status post Swan-Ganz catheter placement and Dobbhoff tube. COMPARISON: Multiple prior chest radiographs, most recently earlier in the same day at 9:00 a.m. FINDINGS: Single portable AP chest radiograph was obtained. There is a Dobbhoff tube which coils in the mid esophagus. A Swan-Ganz catheter has been inserted with its tip at the right main pulmonary artery. There is stable position of an endotracheal tube and an orogastric tube and a right PICC line in the mid SVC. Stable appearance of the small left pleural effusion and stable cardiomegaly. IMPRESSION: Dobbhoff tube which is coiled in the mid thorax, appropriate position of Swan-Ganz catheter. Findings were communicated with Dr. ___ at 1:30 p.m. on ___. Note 21: HISTORY: Dobbhoff placement. FINDINGS: In comparison with study of ___, the Dobbhoff tube now extends to distal stomach. Otherwise, little change. Note 22: HISTORY: Dobbhoff placement. FINDINGS: In comparison with the study of ___, the tip of the Dobbhoff tube is in the distal stomach. Little overall change in the appearance of the heart and lungs with extensive opacification in the retrocardiac region. Note 23: HISTORY: Dobbhoff repositioned. FINDINGS: In comparison with the earlier study of this date, the tip of the Dobbhoff tube has been pulled back to the mid body of the stomach. Remainder of the study is unchanged. Note 24: CHEST HISTORY: Shortness of breath. Bacteremia and endocarditis. Single AP view of the chest performed at 8:35 a.m. is compared to prior study performed ___ at 1815. Tip of the feeding tube remains in the stomach. Tip of the right-sided PICC line is at the cavoatrial junction. There is blunting of the right costophrenic angle essentially unchanged. Increased opacity at the right lung base is most consistent with atelectasis. The heart is enlarged. There is increased opacity in the left costophrenic angle likely secondary to increasing pleural effusion. There is increased opacity in the retrocardiac area which is unchanged. IMPRESSION: Tubes and lines in adequate position. Compared to the prior study there is likely increased opacity in the left costophrenic angle consistent with increasing pleural effusion. Dense retrocardiac opacity persists. New right lower lobe opacity most consistent with atelectasis. Note 25: INDICATION: An ___ male with bacteremia and endocarditis, as well as left upper extremity edema. Evaluate for DVT. COMPARISON: None. FINDINGS: Grayscale and color sonographic imaging of the left internal jugular, subclavian, axillary, brachial, basilic, and cephalic veins was performed. The veins demonstrate normal compressibility, flow, and augmentation. There is no echogenic intraluminal thrombus seen. The contralateral subclavian vein was interrogated for comparison purposes, demonstrating symmetric respiratory phasicity. IMPRESSION: No left upper extremity DVT. Note 26: AP CHEST, 9:56 A.M., ___ HISTORY: An ___ man status post MVR. Dobbhoff tube placement. IMPRESSION: AP chest compared to ___ through ___: Feeding tube ends in the region of the pylorus. Chest is otherwise essentially unchanged over several days, including the chronically enlarged heart, collapsed left lower lobe, small left pleural effusion, borderline edema in the right upper lobe, and persistent right lower lobe consolidation concerning for pneumonia. Right PIC line ends in the low SVC. No pneumothorax. Note 27: CHEST RADIOGRAPH INDICATION: Assessment of Swan-Ganz catheter. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the patient was intubated. The tip of the endotracheal tube projects 4.4 cm above the carina. The patient also received a left subclavian venous introduction sheath. There is no safe evidence of pneumothorax. The nasogastric tube has been exchanged, the tip of the current tube projects over the middle parts of the stomach. Unchanged course and position of the right PICC line. Unchanged size of the cardiac silhouette. Unchanged small left pleural effusion and retrocardiac atelectasis. Improved ventilation of the right lung with minimal right basal pleural effusion. No newly occurred focal parenchymal opacities. Note 28: ULTRASOUND ABDOMEN INDICATION: Post-mitral valve replacement, right upper quadrant pain and hematocrit drop. COMPARISON: Comparison is made with prior ultrasound performed ___. FINDINGS: Technically diffivult examination due to patient body habitus. There is a large shadowing calculus identified within the neck of the gallbladder. There is extensive gallbladder wall thickening and pericholecystic fluid with gallbladder distension and sludge in keeping with acute cholecystitis. Common bile duct measures 7 mm. No intrahepatic biliary dilatation is identified. There is a large collection identified in the retroperitoneum inferior to the right lobe of the liver and posterior to the right kidney it measures 7.8 x 18 cm in length and contains internal echogenicity and septations consistent with a retroperitoneal hematoma on the right side. A 1.4-cm simple cyst is identified mainly in the right lobe of the liver. IMPRESSION: 1. Acute cholecystitis. 2. Large retroperitoneal hematoma (right). Case discussed with ___ in person. Note 29: INDICATION: ___ male status post mitral valve replacement with question of mesenteric ischemia. COMPARISON: CT of the chest without IV contrast from ___. TECHNIQUE: 64-row MDCT obtained of the abdomen and pelvis with images from the lung bases through to the proximal femora with only oral contrast. FINDINGS: There are bilateral pleural effusions in the lungs, left greater than right, with associated compressive atelectasis seen in the left lung. There is no pulmonary nodule. In the aerated lung, there is no focal consolidation to suggest pneumonia. There is significant metallic artifact seen at the location of the known replaced mitral valve. Calcifications of the LAD are noted. There is a nasoenteric tube seen with the tip ending in the stomach. Again noted is a 1.3 x 1.2 cm hypodensity in segment V of the liver that is unchanged from prior non-contrast CT. Otherwise, evaluation of the intra-abdominal solid organs is limited without IV contrast. There is ascites seen around the liver. The unenhanced kidneys, pancreas, and adrenals and spleen are unremarkable. There is a large right retroperitoneal hematoma tracking along the psoas measuring 11 cm AP x 11 cm craniocaudal x 15 cm in transverse direction with a hematocrit level suggesting acute on subacute evolving hematoma. There is no evidence of free air within the abdomen. There are numerous gallstones within the gallbladder as well as a distended gallbladder with gallbladder wall edema. There is no evidence of intra- abdominal wall distention or bowel wall thickening. Again seen is a 2.8-cm hypodensity within the right kidney that is appreciated on the earlier chest CT and is unchanged in size. Note is made of atherosclerotic calcification seen in the infrarenal aorta as well as the common iliacs. CT PELVIS WITH ORAL CONTRAST: Patient has a catheter within the right femoral artery with no associated hematoma. There also is a catheter seen within the left femoral vein with the tip ending proximally at the bifurcation of the external and internal iliac. There is an intrapelvic fluid collection measuring roughly 7.6 x 9.3 cm (300B, 42) with what appears to be a layering hematocrit effect. The bladder is decompressed with a Foley catheter in place. The prostate is unremarkable. There is diffuse stranding of the subcutaneous tissues consistent with anasarca. OSSEOUS STRUCTURES: There is a levocurvature of the thoracic spine. There are no fractures or suspicious lytic or blastic lesions to suggest metastatic disease. IMPRESSION: 1. Large retroperitoneal hematoma with layering hematocrit suggesting acute on chronic component to this evolving hematoma. 2. Free fluid within the rectovesicular space with question of layering hematocrit effect suggestive of a possible bowel perforation. 3. Distended gallbladder with gallbladder wall edema concerning for acute cholecystitis in the appropriate clinical setting 4. No definitive bowel wall distention or bowel wall thickening to suggest ischemia; however, as mentioned, there is an intrapelvic fluid collection. Note 30: PROCEDURE: Ultrasound-guided percutaneous cholecystostomy. INDICATION: Acute cholecystitis, sepsis, and hypotension. COMPARISON: Comparison has been made with prior CT, ___ and ultrasound, ___. PROCEDURE: The risks, benefits, and alternatives to the procedure were explained to the patient's wife, and verbal consent was obtained over the phone. The procedure was performed in the cardiovascular ICU with portable ultrasound. Under ultrasound guidance, an entrance site was selected, and the skin was prepped and draped in the usual sterile fashion. A preprocedure timeout was performed using a single patient identifier. 1% buffered lidocaine was instilled for local anesthesia. US of the gallbladder demonstrated sludge and gallstones with gallbladder wall thickening and pericholecystic fluid.There was also small trace of ___ ascites. An 8fr catheter was inserted under ultrasound guidance via a subcostal approach. 65 cc of dark reddish bile was aspirated, and catheter was left on free drainage at the end of the procedure. Sample of bile was sent to microbiology for culture and sensitivity. The patient was ventilated and sedated in the ICU. Patient tolerated the procedure well, and there were no immediate complications. Dr. ___ attending radiologist, and Dr. ___, the fellow, were present throughout the procedure. Post-procedure instructions were written in the ___ medical record. IMPRESSION: Technically successful ultrasound-guided percutaneous cholecystostomy. Note 31: AP CHEST, 11:46 A.M., ___ HISTORY: MVR. Line placement. IMPRESSION: AP chest compared to ___: Tip of the new left subclavian line ends alongside the right PICC line at the junction of the brachiocephalic veins. Mild interstitial edema is new accompanied by increase in heart size, still at the upper limits of normal and new small left pleural effusion. There is no pneumothorax. Left lower lobe is still airless. ET tube is in standard placement. No pneumothorax. Note 32: AP CHEST, 1:10 P.M. ON ___ HISTORY: ___ man after mitral valve repair. IMPRESSION: AP chest compared to ___ through ___: Mild pulmonary edema present on ___ has improved. Left lower lobe collapse and moderate left pleural effusion has not. There is no pneumothorax. Mild-to-moderate cardiomegaly is stable, and mediastinum is not widened. ET tube is in standard position, nasogastric tube ends in the distal stomach, and a left subclavian line ends alongside a right PIC line in the upper SVC. No pneumothorax. Note 33: AP CHEST 6 P.M. ON ___ HISTORY: Renal failure. IMPRESSION: Left subclavian line ends at the junction of brachiocephalic veins, dual-channel right internal jugular line ends in the low SVC. No pneumothorax, new mediastinal widening or right pleural effusion. Pulmonary vascular engorgement has improved, previous mild interstitial edema has resolved, left lower lobe atelectasis has improved and previous moderate left pleural effusion has decreased. Heart size top normal. ET tube in standard placement. Nasogastric tube ends in the distal stomach. Right upper quadrant drainage catheter noted but cannot be localized on this single frontal view. Note 34: HISTORY: Postoperative complications with new hemoptysis. FINDINGS: In comparison with the study of ___, there is continued vascular engorgement with some left basilar atelectasis and pleural effusion in a patient with dense calcification in the mitral region. Tracheostomy tube is in good position as are the central catheters. Note 35: LIVER ULTRASOUND INDICATION: History of prior percutaneous cholecystostomy with rising bilirubin. COMPARISON: Ultrasound percutaneous cholecystostomy ___, CT abdomen and pelvis ___, ultrasound abdomen ___. FINDINGS: The cholecystostomy tube is seen within the gallbladder lumen. The gallbladder is decompressed with extensive gallbladder wall thickening noted. A small amount of pericholecystic fluid is also noted. Multiple shadowing gallstones and sludge noted within the gallbladder. Tiny trace of perihepatic free fluid is noted. There is no evidence of intrahepatic biliary dilatation. The common bile duct measures 5 mm. There is normal liver echotexture. There is a 1.0 x 0.9 x 1.3 cm hypoechoic lesion identified within the right lobe of the liver, likely simple hepatic cysts. Pancreas visualized in the midline though body and tail are not visualized in their entirety. Spleen is normal in caliber. Note is made of atelectasis / consolidation of the left lower lobe with associated small left effusion. IMPRESSION: 1. Decompressed gallbladder with cholecystostomy tube noted within the gallbladder wall lumen. Sludge and stones noted within the lumen.Trace pericholecystic fluid noted. 2. No intrahepatic or extrahepatic biliary dilatation. 3. Simple hepatic cyst. Note 36: INDICATION: ___ man with history of renal failure, for dialysis. PHYSICIAN: Dr. ___, the attending radiologist, performed the procedure. PROCEDURES: 1. Initial fluoroscopic spot image. 2. Placement of a 23-cm tip-to-cuff tunneled dialysis catheter via the right internal jugular vein. 3. Post-placement fluoroscopic spot image. MEDICATIONS: The patient received moderate sedation with two divided doses of 15 mcg of fentanyl IV and two divided doses of 1 mg IV Versed throughout the intraservice time of 20 minutes, during which continuous hemodynamic monitoring was performed by a trained radiology nurse. Additionally, 1% local lidocaine was administered. PROCEDURE: Prior to initiation of procedure, written informed consent was obtained and a preprocedure timeout was performed. The right upper neck was prepped and draped in a sterile manner. Under ultrasound guidance, micropuncture access was obtained into the right internal jugular vein. Pre- and post-access hard copy ultrasound images were obtained are on file. A site along the anterior chest wall was then selected, and after local lidocaine administration was performed, a tunneling device was used to tunnel the catheter to the venous access site. Next, sequential fascial dilatation of the venotomy was performed over the existing guidewire, and a peel-away sheath was placed. The catheter was advanced through the peel-away sheath, such that the tip was positioned in the right atrium. The venous access site was sutured with ___ Vicryl suture. Catheter was secured with ___ silk suture. The catheter aspirated and flushed well and was in good location on the final fluoroscopic spot image. The patient tolerated the procedure well. IMPRESSION: Successful placement of a tunneled hemodialysis catheter (23-cm tip-to-cuff) via the right internal jugular vein with tip in the right atrium. The catheter is ready to use. Note 37: INDICATION: ___ male with bacteremia and poor aeration on the left side. COMPARISON: ___. TECHNIQUE: Single AP radiograph of the chest was obtained with the patient in the upright position. FINDINGS: There is complete opacification of the left lung with shift of mediastinal structures towards the left, consistent with collapse. Evaluation of the left pleural effusion cannot be performed in this setting. The right lung demonstrates pulmonary vascular congestion and a small effusion. Extensive calcification of the mitral annulus and mitral valve repair are visualized. The tracheostomy tube is visualized in similar position compared to prior. There is a right-sided central catheter with tip in the right atrium. Sternal wires are noted. There has been interval removal of the left subclavian catheter. A right-sided PICC catheter is seen with tip at the confluence of the brachiocephalic veins. Severe scoliosis is seen. IMPRESSION: Left lung collapse. These findings were discussed with ___, NP by Dr. ___ by telephone at 10:40 a.m. on ___. Note 38: INDICATION: ___ male with bacteremia and left lung collapse, now status post bronchoscopy. ___ at 08:00 a.m. TECHNIQUE: Single AP radiograph of the chest was obtained. FINDINGS: The lung apices are not included in this view. Compared to most recent prior, there has been partial re-expansion of the left upper lung. There is persistent opacification of the left lower hemithorax. There is a left pleural effusion which is incompletely evaluated in the setting of partial left lung collapse. The right lung is unchanged. Tracheostomy tube, central line with tip in the right atrium, mitral annular calcification and sternal wires are again seen. IMPRESSION: Partial reexpansion of the left upper lung with persistent collapse of the left lower lobe. These findings were discussed with ___, NP by Dr. ___ by telephone at approximately 10:45 a.m. on ___. | False | 0 | Based on the following medical notes, please predict whether the patient described is likely to have Acute Respiratory Distress Syndrome (ARDS). Your prediction should be either 'true' if ARDS is likely, or 'false' if it is not likely. |

The other aspect worth noting is the average number of characters in each of the three columns instruction, input and output in the dataset. Typically, every 3-4 characters maps to a token (the basic building blocks that language models use to understand and analyze text data), and large language models have a limit on the number of tokens they can take as input.

```
# Calculating the length of each cell in each column
df['num_characters_instruction'] = df['instruction'].apply(lambda x: len(x))
df['num_characters_notes'] = df['notes'].apply(lambda x: len(x))
df['num_characters_output'] = df['output'].apply(lambda x: len(x))

# Show Distribution
df.hist(column=['num_characters_instruction', 'num_characters_notes', 'num_characters_output'])

# Calculating the average
average_chars_instruction = df['num_characters_instruction'].median()
average_chars_notes = df['num_characters_notes'].median()
average_chars_output = df['num_characters_output'].median()

print(f'Average number of tokens in the instruction column: {(average_chars_instruction / 3):.0f}')
print(f'Average number of tokens in the notes column: {(average_chars_notes / 3):.0f}')
print(f'Average number of tokens in the output column: {(average_chars_output / 3):.0f}', end="\n\n")
```

Average number of tokens in the instruction column: 78
Average number of tokens in the notes column: 1738
Average number of tokens in the output column: 2

The maximum context length for the base LLaMA-2 model is 4096 tokens. However, since we are using a free GPU, the GPU Memory limits the token length to around 400. We need to truncate the medical notes to avoid out-of-memory issue. A dummy method is to the get the first 400-78(instruction token)-2(output token)=320 tokens(1260 characters) in the notes and drop the rest. Later, you will need to design your own method to truncate the notes. If you get an CUDA OUT OF MEMORY error during fine-tuning, you may come back to this cell and use a more strict token limit. You can monitor the GPU RAM usage on the top right.

```
TOKEN_LIMIT = 400

characters_allowed = int((TOKEN_LIMIT - average_chars_instruction/3 - average_chars_output/3)*3)

df['input'] = df['notes'].apply(lambda x: x[0:min(len(x), characters_allowed)])

df.head(2)

```
| index | notes | output | split | Instrunction | index | notes | output | split | Instrunction | num_characters_instruction | num_characters_notes | num_characters_output | Input |
|---|---|---| --- | --- | ---|---|---| --- | --- | ---|---|---| --- | 
| 1 | Note 1: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with chest tubes// Assess placement Assess placement IMPRESSION: Comparison to ___. The bilateral chest tubes are in stable position. Stable extent and severity of the pre-existing bilateral parenchymal opacities, right more than left, no evidence of pneumothorax. Stable mild cardiomegaly. Note 2: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with PTX s/p bilateral chest tubes// Assess for placement, remaining air Assess for placement, remaining air IMPRESSION: Comparison to ___. The patient has been extubated. The feeding tube and the bilateral chest tubes are in stable position. A right pneumothorax is approximately 1 cm in diameter. On the left, no pneumothorax is currently visualized. Slightly increasing extent of a retrocardiac atelectasis. Otherwise unchanged appearance of the lung parenchyma. Stable mild cardiomegaly. No evidence of tension. Note 3: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with bilateral PTX and chest tube, removed left tube, right with persistent PTX.// Assess for change in right sided PTX and any new PTX after left chest tube removal? Assess for change in right sided PTX and any new PTX after left chest tube removal? IMPRESSION: After removal of the left chest tube, a 1 cm left apical pneumothorax has reoccurred. There is no evidence of tension. On the right, despite the present chest tube, a 1 cm right apical pneumothorax persists. Again, there is no evidence of tension. Stable parenchymal opacities. Stable moderate cardiomegaly. Note 4: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with bilateral PTX, right chest tube in place, left removed// stability of bilateral PTX IMPRESSION: In comparison with the study of ___, after removal of the left chest tube any residual pneumothorax is quite small. Right chest tube remains in place with little change change in the small apical pneumothorax. The left hemidiaphragmatic contour is more sharply seen, suggesting decreasing volume loss and pleural fluid, with similar changes on the right. A similar appearance could merely reflect a more upright position of the patient. Small amount of subcutaneous gas is again seen along the right lateral chest wall. Note 5: EXAMINATION: Chest radiograph, portable AP upright. INDICATION: Query pneumothorax. Chest tube in place. COMPARISON: Prior study from earlier on the same day. FINDINGS: Right-sided chest tube and PICC lines appear unchanged in position. Previously, pneumothorax was very small, measuring only up to 5 mm in with at the lung apex but now up to 23 mm. There is probably a small pleural fluid component, and increased opacification and volume loss at the right lung base suggest atelectasis. Streaky basilar opacities on the left also suggests minor atelectasis with a trace pleural effusion. IMPRESSION: Increasing right pneumothorax, now mild to moderate. NOTIFICATION: The findings were discussed with Dr. ___ , M.D. by ___, M.D. on the telephone on ___ at 5:53 pm, 2 minutes after discovery of the findings. Note 6: EXAMINATION: Chest radiograph, portable AP upright INDICATION: Follow-up pneumothorax. COMPARISON: Earlier on the same evening. FINDINGS: There is a tiny right apical pneumothorax, considerably decreased and largely resolve. Chest tube and PICC line appear unchanged. Right basilar atelectasis has largely resolved. There is similar minor left basilar atelectasis. Trace pleural fluid is probably unchanged bilaterally. Trace subcutaneous emphysema appears unchanged bilaterally. IMPRESSION: Marked decrease in right pneumothorax. Note 7: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with right chest tube in place, has continued to be on low wall suction// ?interval change IMPRESSION: In comparison with the study of the ___, with the right chest tube on suction, there is little change in the small apical pneumothorax. A the minimal subcutaneous gas remains along the right lateral chest wall. The opacification at the bases most likely represents residual atelectatic changes. Note 8: EXAMINATION: Chest radiograph INDICATION: ___ year old man with bilateral PTX// assess for interval change. PLEASE OBTAIN AT 1230 TECHNIQUE: Chest PA and lateral COMPARISON: Multiple prior chest radiographs, most recently ordered today. FINDINGS: Unchanged position of right-sided chest tube and PICC line. Minimal change in bilateral pneumothorax. No pleural effusions. Mild bibasilar atelectasis. Heart size remains normal with unremarkable cardiomediastinal silhouette. IMPRESSION: No significant interval change of bilateral small apical pneumothorax. Note 9: INDICATION: ___ year old man with acute hypoxic respiratory failure/PNA c/b bilateral pneumothoraces s/p bilateral chest tubes. Left removed ___, right ___// evaluate for stability/improvement in bilateral pneumothoraces TECHNIQUE: Chest PA and lateral COMPARISON: ___ IMPRESSION: The small right-sided pneumothorax is unchanged. Right-sided chest tube has been removed in the interim. Right-sided PICC line projects to the distal SVC. There is minimal subsegmental atelectasis in both lung bases. Cardiomediastinal silhouette is stable. Trace left pneumothorax has slightly improved Note 10: INDICATION: ___ year old man with hypoxic resp failure c/b bilateral PTX now s/p CT removal// pls evaluate for interval changes/improvement in PTX TECHNIQUE: Chest PA and lateral COMPARISON: Chest x-ray ___ FINDINGS: Redemonstration of a right-sided PICC, which terminates in the mid SVC. Interval reduction in bilateral chest wall subcutaneous emphysema. Miniscule residual of pleural air, if any, right apex and left base. Tiny left pleural effusion has almost resolved. Small area of atelectasis or persistent consolidation in left lung laterally, developed between ___ and ___. Basal linear atelectasis is improving. Lungs otherwise essentially clear. Cardiomediastinal silhouette is normal. Bilateral hila are unremarkable. IMPRESSION: 1. Miniscule if any pneumothorax, right apex, left base. 2. Small area of atelectasis or persistent consolidation in the left lower lung. 3. Interval near resolution tiny left pleural effusion and improved bibasilar atelectasis. Note 11: EXAMINATION: CHEST (PORTABLE AP) INDICATION: History: ___ with Please confirm chest tube placement// Please confirm chest tube placement TECHNIQUE: Single frontal view of the chest COMPARISON: None FINDINGS: Endotracheal tube terminates 6.2 cm above the carina. Enteric tube courses below the diaphragm, out of the field of view. There are bilateral chest tubes. There are extensive bilateral pulmonary opacities. Blunting of the costophrenic angles may be due to pleural effusions. Cardiac and mediastinal silhouettes are grossly unremarkable. IMPRESSION: Endotracheal tube terminates 6.2 cm above the carina. Enteric tube courses below the diaphragm, out of the field of view. Extensive bilateral nodular pulmonary opacities with no prior available for comparison. Findings may be due to extensive multifocal pneumonia, ARDS, underlying malignancy not excluded. Blunting of the costophrenic angles could be due to pleural effusions and/or atelectasis. Note 12: EXAMINATION: CT torso with intravenous contrast. INDICATION: ___ with severe sepsis and ARDS// e/o cholangitis? intraabdominal processes TECHNIQUE: MDCT axial images were acquired through the chest, abdomen and pelvis following intravenous contrast administration with split bolus technique. Oral contrast was not administered.Coronal and sagittal reformations were performed and reviewed on PACS. DOSE: Total DLP (Body) = 609 mGy-cm. COMPARISON: None. FINDINGS: CHEST: Endotracheal tube appears to terminate 4.3 cm above the carina. The NG tube terminates in the stomach. Bilateral chest tubes terminate at the apex. Extensive bilateral pulmonary consolidations most notably in the lower lungs consistent with multifocal pneumonia. Thoracic aorta is normal in course and caliber. The main pulmonary artery and central branches are patent. No lymphadenopathy is seen. Heart is normal in size and shape. No pleural or pericardial effusion is seen. There is no residual pneumothorax. ABDOMEN: Anomalous anatomy in the upper abdomen likely reflect congenital absence of the right portal vein with resultant atrophy. Small bowel is seen clustered in the right subdiaphragmatic space. The liver measures 19 cm in length. No discrete focal liver lesion seen. Main portal vein is patent. The gallbladder contains a gallstone and is decompressed. There is no biliary ductal dilation. The spleen is mildly enlarged measuring 14 cm in length. The kidneys appear normal though both appear slightly displaced superiorly. Adrenal glands appear unremarkable. The abdominal aorta is normal in course and caliber. Anomalous vascular anatomy noted a common trunk from the abdominal aorta giving rise to the right renal artery, celiac trunk and SMA. The pancreas appears grossly unremarkable. The stomach is unremarkable. The duodenum follows an unusual course extending superiorly into the right upper abdomen. The small bowel is difficult to trace beyond the proximal duodenum in the absence of oral contrast. Malrotation is difficult to exclude. No lymphadenopathy. There is a small volume free fluid. No free air. PELVIS: No signs of ileus or obstruction. The appendix is normal. The colon is mostly decompressed. Small volume free fluid in the pelvis is noted. The urinary bladder is decompressed around a Foley catheter. There is no pelvic sidewall or inguinal adenopathy. A right common femoral venous catheter is in place. BONES: There is no acute fracture. No focal suspicious osseous abnormality. SOFT TISSUES: The abdominal and pelvic wall is within normal limits. IMPRESSION: 1. Extensive bilateral pulmonary airspace opacities most confluent in the lower lobes concerning for multifocal pneumonia. 2. Chest tubes in place without residual pneumothorax. 3. ET and NG tubes positioned appropriately. 4. Small volume ascites with enlarged liver measuring 18.5 cm. Splenomegaly also noted. Clinical correlation advised. 5. Anomalous anatomy in the abdomen likely reflecting congenital absence of the right portal vein and resultant right hepatic atrophy. 6. Anomalous branching anatomy from the abdominal aorta as described. 7. Small-bowel clustered in the right subdiaphragmatic space with possible small bowel malrotation. Note 13: EXAMINATION: DX CHEST PORT LINE/TUBE PLCMT 1 EXAM INDICATION: ___ year old man with ET tube, just advanced// check placement check placement IMPRESSION: Bilateral chest tube is present. NG tube tip is in the stomach. Multifocal extensive consolidations are similar to previous examination. Minimal right apical pneumothorax cannot be excluded. Note 14: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with resp failure on vent// Eval for changes, ETT placement Eval for changes, ETT placement IMPRESSION: Comparison to ___. The monitoring and support devices, including the bilateral chest tubes, are in stable correct position. The current radiograph shows no evidence for the presence of a pneumothorax. The parenchymal opacities, right more than left, are stable. No new parenchymal changes are noted. Stable moderate cardiomegaly. Note 15: INDICATION: ___ year old man with ARDS, PNA, pneumothraces s/p b/l chest tube. Increased tachacardia and hypoxemia// Pneumothorax, worsening effusions/infiltrates TECHNIQUE: AP portable chest radiograph COMPARISON: ___ FINDINGS: The tip of the endotracheal tube projects over the midthoracic trachea. A feeding tube extends to the stomach. Bilateral chest tubes are present. There is no discrete pneumothorax identified. Re-demonstrated are bilateral parenchymal opacities, increased in the right mid to lower lung and left base. The size of the cardiac silhouette is unchanged. IMPRESSION: Increasing bilateral parenchymal opacities. No discrete pneumothorax identified. Note 16: EXAMINATION: LIVER OR GALLBLADDER US (SINGLE ORGAN) INDICATION: ___ year old man with ards, persistent LFT derangements, hepatomegaly// Biliary dilation, assess for any obvious hepatic abnormalities TECHNIQUE: Grey scale and color Doppler ultrasound images of the abdomen were obtained. COMPARISON: CT abdomen and pelvis dated ___. FINDINGS: LIVER: The hepatic parenchyma appears within normal limits. The contour of the liver is smooth. There is no focal liver mass. The main portal vein is patent with hepatopetal flow. There is no ascites. BILE DUCTS: There is no intrahepatic biliary dilation. CHD: 3 mm GALLBLADDER: Cholelithiasis without gallbladder wall thickening. PANCREAS: The pancreas is not well visualized, largely obscured by overlying bowel gas. SPLEEN: Normal echogenicity. Spleen length: 9.7 cm KIDNEYS: Limited views of the kidneys show no hydronephrosis. RETROPERITONEUM: The visualized portions of aorta and IVC are within normal limits. IMPRESSION: 1. Normal appearing liver without biliary dilatation. 2. Cholelithiasis, but no evidence of acute cholecystitis. Note 17: EXAMINATION: CHEST PORT. LINE PLACEMENT INDICATION: ___ year old man with new R PICC// R DL Power PICC 37cm ___ ___ Contact name: ___: ___ R DL Power PICC 37cm ___ ___ IMPRESSION: ET tube tip is 5 cm above the carina. NG tube tip is in the stomach. Right PICC line tip is at the level of lower SVC. Right chest tube is in place. Left chest tube is in place. Unchanged are multifocal consolidations that are better appreciated on the recent chest CT from ___.	False	0	Based on the following medical notes, please predict whether the patient described is likely to have Acute Respiratory Distress Syndrome (ARDS). Your prediction should be either 'true' if ARDS is likely, or 'false' if it is not likely.	235	13852	5	Note 1: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with chest tubes// Assess placement Assess placement IMPRESSION: Comparison to ___. The bilateral chest tubes are in stable position. Stable extent and severity of the pre-existing bilateral parenchymal opacities, right more than left, no evidence of pneumothorax. Stable mild cardiomegaly. Note 2: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with PTX s/p bilateral chest tubes// Assess for placement, remaining air Assess for placement, remaining air IMPRESSION: Comparison to ___. The patient has been extubated. The feeding tube and the bilateral chest tubes are in stable position. A right pneumothorax is approximately 1 cm in diameter. On the left, no pneumothorax is currently visualized. Slightly increasing extent of a retrocardiac atelectasis. Otherwise unchanged appearance of the lung parenchyma. Stable mild cardiomegaly. No evidence of tension. Note 3: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with bilateral PTX and chest tube, removed left tube, right with persistent PTX.// Assess for change in right sided PTX and any new PTX after left chest tube removal? Assess for change in right sided PTX and any new PTX after left chest tube removal? IMPRESSION: After removal of the left chest tube, a 1 cm left apical pneumothorax has reoccurred. There is no evidence of tension. On the right, despite the present chest tube, a 1 cm right apical pneumothorax persists. Again, there is no evidence of tension. Stable parenchymal opacities. Stable moderate cardiomegaly. Note 4: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with bilateral PTX, right chest tube in place, left removed// stability of bilateral PTX IMPRESSION: In comparison with the study of ___, after removal of the left chest tube any residual pneumothorax is quite small. Right chest tube remains in place with little change change in the small apical pneumothorax. The left hemidiaphragmatic contour is more sharply seen, suggesting decreasing volume loss and pleural fluid, with similar changes on the right. A similar appearance could merely reflect a more upright position of the patient. Small amount of subcutaneous gas is again seen along the right lateral chest wall. Note 5: EXAMINATION: Chest radiograph, portable AP upright. INDICATION: Query pneumothorax. Chest tube in place. COMPARISON: Prior study from earlier on the same day. FINDINGS: Right-sided chest tube and PICC lines appear unchanged in position. Previously, pneumothorax was very small, measuring only up to 5 mm in with at the lung apex but now up to 23 mm. There is probably a small pleural fluid component, and increased opacification and volume loss at the right lung base suggest atelectasis. Streaky basilar opacities on the left also suggests minor atelectasis with a trace pleural effusion. IMPRESSION: Increasing right pneumothorax, now mild to moderate. NOTIFICATION: The findings were discussed with Dr. ___ , M.D. by ___, M.D. on the telephone on ___ at 5:53 pm, 2 minutes after discovery of the findings. Note 6: EXAMINATION: Chest radiograph, portable AP upright INDICATION: Follow-up pneumothorax. COMPARISON: Earlier on the same evening. FINDINGS: There is a tiny right apical pneumothorax, considerably decreased and largely resolve. Chest tube and PICC line appear unchanged. Right basilar atelectasis has largely resolved. There is similar minor left basilar atelectasis. Trace pleural fluid is probably unchanged bilaterally. Trace subcutaneous emphysema appears unchanged bilaterally. IMPRESSION: Marked decrease in right pneumothorax. Note 7: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with right chest tube in place, has continued to be on low wall suction// ?interval change IMPRESSION: In comparison with the study of the ___, with the right chest tube on suction, there is little change in the small apical pneumothorax. A the minimal subcutaneous gas remains along the right lateral chest wall. The opacification at the bases most likely represents residual atelectatic changes. Note 8: EXAMINATION: Chest radiograph INDICATION: ___ year old man with bilateral PTX// assess for interval change. PLEASE OBTAIN AT 1230 TECHNIQUE: Chest PA and lateral COMPARISON: Multiple prior chest radiographs, most recently ordered today. FINDINGS: Unchanged position of right-sided chest tube and PICC line. Minimal change in bilateral pneumothorax. No pleural effusions. Mild bibasilar atelectasis. Heart size remains normal with unremarkable cardiomediastinal silhouette. IMPRESSION: No significant interval change of bilateral small apical pneumothorax. Note 9: INDICATION: ___ year old man with acute hypoxic respiratory failure/PNA c/b bilateral pneumothoraces s/p bilateral chest tubes. Left removed ___, right ___// evaluate for stability/improvement in bilateral pneumothoraces TECHNIQUE: Chest PA and lateral COMPARISON: ___ IMPRESSION: The small right-sided pneumothorax is unchanged. Right-sided chest tube has been removed in the interim. Right-sided PICC line projects to the distal SVC. There is minimal subsegmental atelectasis in both lung bases. Cardiomediastinal silhouette is stable. Trace left pneumothorax has slightly improved Note 10: INDICATION: ___ year old man with hypoxic resp failure c/b bilateral PTX now s/p CT removal// pls evaluate for interval changes/improvement in PTX TECHNIQUE: Chest PA and lateral COMPARISON: Chest x-ray ___ FINDINGS: Redemonstration of a right-sided PICC, which terminates in the mid SVC. Interval reduction in bilateral chest wall subcutaneous emphysema. Miniscule residual of pleural air, if any, right apex and left base. Tiny left pleural effusion has almost resolved. Small area of atelectasis or persistent consolidation in left lung laterally, developed between ___ and ___. Basal linear atelectasis is improving. Lungs otherwise essentially clear. Cardiomediastinal silhouette is normal. Bilateral hila are unremarkable. IMPRESSION: 1. Miniscule if any pneumothorax, right apex, left base. 2. Small area of atelectasis or persistent consolidation in the left lower lung. 3. Interval near resolution tiny left pleural effusion and improved bibasilar atelectasis. Note 11: EXAMINATION: CHEST (PORTABLE AP) INDICATION: History: ___ with Please confirm chest tube placement// Please confirm chest tube placement TECHNIQUE: Single frontal view of the chest COMPARISON: None FINDINGS: Endotracheal tube terminates 6.2 cm above the carina. Enteric tube courses below the diaphragm, out of the field of view. There are bilateral chest tubes. There are extensive bilateral pulmonary opacities. Blunting of the costophrenic angles may be due to pleural effusions. Cardiac and mediastinal silhouettes are grossly unremarkable. IMPRESSION: Endotracheal tube terminates 6.2 cm above the carina. Enteric tube courses below the diaphragm, out of the field of view. Extensive bilateral nodular pulmonary opacities with no prior available for comparison. Findings may be due to extensive multifocal pneumonia, ARDS, underlying malignancy not excluded. Blunting of the costophrenic angles could be due to pleural effusions and/or atelectasis. Note 12: EXAMINATION: CT torso with intravenous contrast. INDICATION: ___ with severe sepsis and ARDS// e/o cholangitis? intraabdominal processes TECHNIQUE: MDCT axial images were acquired through the chest, abdomen and pelvis following intravenous contrast administration with split bolus technique. Oral contrast was not administered.Coronal and sagittal reformations were performed and reviewed on PACS. DOSE: Total DLP (Body) = 609 mGy-cm. COMPARISON: None. FINDINGS: CHEST: Endotracheal tube appears to terminate 4.3 cm above the carina. The NG tube terminates in the stomach. Bilateral chest tubes terminate at the apex. Extensive bilateral pulmonary consolidations most notably in the lower lungs consistent with multifocal pneumonia. Thoracic aorta is normal in course and caliber. The main pulmonary artery and central branches are patent. No lymphadenopathy is seen. Heart is normal in size and shape. No pleural or pericardial effusion is seen. There is no residual pneumothorax. ABDOMEN: Anomalous anatomy in the upper abdomen likely reflect congenital absence of the right portal vein with resultant atrophy. Small bowel is seen clustered in the right subdiaphragmatic space. The liver measures 19 cm in length. No discrete focal liver lesion seen. Main portal vein is patent. The gallbladder contains a gallstone and is decompressed. There is no biliary ductal dilation. The spleen is mildly enlarged measuring 14 cm in length. The kidneys appear normal though both appear slightly displaced superiorly. Adrenal glands appear unremarkable. The abdominal aorta is normal in course and caliber. Anomalous vascular anatomy noted a common trunk from the abdominal aorta giving rise to the right renal artery, celiac trunk and SMA. The pancreas appears grossly unremarkable. The stomach is unremarkable. The duodenum follows an unusual course extending superiorly into the right upper abdomen. The small bowel is difficult to trace beyond the proximal duodenum in the absence of oral contrast. Malrotation is difficult to exclude. No lymphadenopathy. There is a small volume free fluid. No free air. PELVIS: No signs of ileus or obstruction. The appendix is normal. The colon is mostly decompressed. Small volume free fluid in the pelvis is noted. The urinary bladder is decompressed around a Foley catheter. There is no pelvic sidewall or inguinal adenopathy. A right common femoral venous catheter is in place. BONES: There is no acute fracture. No focal suspicious osseous abnormality. SOFT TISSUES: The abdominal and pelvic wall is within normal limits. IMPRESSION: 1. Extensive bilateral pulmonary airspace opacities most confluent in the lower lobes concerning for multifocal pneumonia. 2. Chest tubes in place without residual pneumothorax. 3. ET and NG tubes positioned appropriately. 4. Small volume ascites with enlarged liver measuring 18.5 cm. Splenomegaly also noted. Clinical correlation advised. 5. Anomalous anatomy in the abdomen likely reflecting congenital absence of the right portal vein and resultant right hepatic atrophy. 6. Anomalous branching anatomy from the abdominal aorta as described. 7. Small-bowel clustered in the right subdiaphragmatic space with possible small bowel malrotation. Note 13: EXAMINATION: DX CHEST PORT LINE/TUBE PLCMT 1 EXAM INDICATION: ___ year old man with ET tube, just advanced// check placement check placement IMPRESSION: Bilateral chest tube is present. NG tube tip is in the stomach. Multifocal extensive consolidations are similar to previous examination. Minimal right apical pneumothorax cannot be excluded. Note 14: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with resp failure on vent// Eval for changes, ETT placement Eval for changes, ETT placement IMPRESSION: Comparison to ___. The monitoring and support devices, including the bilateral chest tubes, are in stable correct position. The current radiograph shows no evidence for the presence of a pneumothorax. The parenchymal opacities, right more than left, are stable. No new parenchymal changes are noted. Stable moderate cardiomegaly. Note 15: INDICATION: ___ year old man with ARDS, PNA, pneumothraces s/p b/l chest tube. Increased tachacardia and hypoxemia// Pneumothorax, worsening effusions/infiltrates TECHNIQUE: AP portable chest radiograph COMPARISON: ___ FINDINGS: The tip of the endotracheal tube projects over the midthoracic trachea. A feeding tube extends to the stomach. Bilateral chest tubes are present. There is no discrete pneumothorax identified. Re-demonstrated are bilateral parenchymal opacities, increased in the right mid to lower lung and left base. The size of the cardiac silhouette is unchanged. IMPRESSION: Increasing bilateral parenchymal opacities. No discrete pneumothorax identified. Note 16: EXAMINATION: LIVER OR GALLBLADDER US (SINGLE ORGAN) INDICATION: ___ year old man with ards, persistent LFT derangements, hepatomegaly// Biliary dilation, assess for any obvious hepatic abnormalities TECHNIQUE: Grey scale and color Doppler ultrasound images of the abdomen were obtained. COMPARISON: CT abdomen and pelvis dated ___. FINDINGS: LIVER: The hepatic parenchyma appears within normal limits. The contour of the liver is smooth. There is no focal liver mass. The main portal vein is patent with hepatopetal flow. There is no ascites. BILE DUCTS: There is no intrahepatic biliary dilation. CHD: 3 mm GALLBLADDER: Cholelithiasis without gallbladder wall thickening. PANCREAS: The pancreas is not well visualized, largely obscured by overlying bowel gas. SPLEEN: Normal echogenicity. Spleen length: 9.7 cm KIDNEYS: Limited views of the kidneys show no hydronephrosis. RETROPERITONEUM: The visualized portions of aorta and IVC are within normal limits. IMPRESSION: 1. Normal appearing liver without biliary dilatation. 2. Cholelithiasis, but no evidence of acute cholecystitis. Note 17: EXAMINATION: CHEST PORT. LINE PLACEMENT INDICATION: ___ year old man with new R PICC// R DL Power PICC 37cm ___ ___ Contact name: ___: ___ R DL Power PICC 37cm ___ ___ IMPRESSION: ET tube tip is 5 cm above the carina. NG tube tip is in the stomach. Right PICC line tip is at the level of lower SVC. Right chest tube is in place. Left chest tube is in place. Unchanged are multifocal consolidations that are better appreciated on the recent chest CT from ___. | False | 0 | 	Based on the following medical notes, please predict whether the patient described is likely to have Acute Respiratory Distress Syndrome (ARDS). Your prediction should be either 'true' if ARDS is likely, or 'false' if it is not likely. | 235 | 13852 | 5 | Note 1: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with chest tubes// Assess placement Assess placement IMPRESSION: Comparison to ___. The bilateral chest tubes are in stable position. Stable extent and severity of the pre-existing bilateral parenchymal opacities, right more than left, no evidence of pneumothorax. Stable mild cardiomegaly. Note 2: EXAMINATION: CHEST (PORTABLE AP) INDICATION: ___ year old man with PTX s/p bilateral chest tubes// Assess for placement, remaining air Assess for placement, remaining air IMPRESSION: Comparison to ___. The patient has been extubated. The feeding tube and the bilateral chest tubes are in stable position. A right pneumothorax is approximately 1 cm in diameter. On the left, no pneumothorax is currently visualized. Slightly increasing extent of a retrocardiac atelectasis. Otherwise unchanged appearance of the lung parenchyma. Stable mild cardiomegaly. No |
| 0 | Note 1: ADDENDUM: In the impression, it states that the free fluid within the rectovesicular space with the layering hematocrit could be a possible bowel perforation. This is incorrect. This fluid collection in the pelvis represents an intraperitoneal extension of the right retroperitoneal hematoma and is not suggestive of bowel perforation. Note 2: REASON FOR EXAMINATION: Hypoxemia. Portable AP chest radiograph was reviewed in comparison to ___. There is interval progression of left lower lobe retrocardiac consolidation currently obscuring the entire left lower lobe and the hemidiaphragm. There is also progression of the right basal consolidation and new right pleural effusion demonstrated. Upper lungs are essentially unchanged. Replaced mitral valve projecting over the significantly calcified mitral annulus is redemonstrated. The right PICC line tip is at the level of low SVC. Note 3: REASON FOR EXAMINATION: Dyspnea. Portable AP chest radiograph was compared to a prior study obtained on ___. Current study demonstrates interval progression of pulmonary edema, interstitial on the top of previously described consolidations and pleural effusion. Note 4: AP CHEST 8:39 A.M. ON ___ HISTORY: An ___ man re-admitted with shortness of breath after MVR. Evaluate for effusion. IMPRESSION: AP chest compared to ___: Moderate to large right pleural effusion not changed appreciably since ___. Left lower lobe remains collapsed accompanied by small to moderate pleural effusion on that side. No pneumothorax. Heart is very large but stable post-operatively and there is no pulmonary edema. Note 5: CHEST RADIOGRAPH INDICATION: Chest tube placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, a right-sided chest tube has been placed. The tube is in correct position. The extent of right pleural effusion has slightly decreased. There is a small extrathoracic air collection at the site of tube insertion. No evidence of right-sided pneumothorax. Unchanged appearance of the cardiac silhouette and of the left lung base, with an extensive left basal atelectasis. No newly appeared focal parenchymal opacity suggesting pneumonia. Note 6: CHEST CT ON ___ HISTORY: Pleural effusion and bacteremia. TECHNIQUE: Multidetector helical scanning of the chest was performed without intravenous contrast agent reconstructed as contiguous 5- and 1.25-mm thick axial and 5-mm thick coronal and parasagittal images, read in conjunction with chest radiographs ___ through ___: FINDINGS: Moderate size nonhemorrhagic bilateral pleural effusion layers posteriorly. A moderate-sized pericardial effusion greatest in the region normally occupied by the left atrial appendage is separated from cardiac chambers by preserved epicardial fat. The attenuation of the pericardial and pleural effusions though below that of frank hemorrhage is above that of simple transudate, which is what one would expect following surgery where fluid has been present for several days and may have small components of blood. There is no radiographic evidence of tamponade physiology. Atelectasis is present in all of the basal segments of the left lower lobe and the posterior basal and some of the lateral basal on the right. Central bronchi are not obstructed. Right apical and anterior pneumothorax is very small. Right pleural tube running up the major fissure to the upper posterior hemithorax, is filled with material. Whether it is occluded can be judged clinically, but conceivably it is isolated from the right pleural effusion. There is no pneumonia. Sternal fragments are closely applied in the body of the sternum, more separated in the manubrium, but the postoperative appearance at all levels is unremarkable and there is no associated fluid collection. Heart is severely enlarged, particularly the atria. Mitral annulus is heavily calcified. IMPRESSION: 1. Moderate bilateral pleural effusions are dependent, with no good evidence for loculation or acute hemorrhage. Differentiation of empyema from persistent, noninfectious, postoperative effusion is not possible radiographically, but should be feasible with image-directed thoracentesis. Pericardial effusion is moderate, with no evidence of tamponade physiology. Right pleural tube is fissural and may be isolated from the right pleural effusion and small right pneumothorax. 2. Bibasilar atelectasis attributable to the pleural effusion. No evidence of pneumonia. 3. Moderate-to-severe cardiomegaly predominantly atrial. MVR within heavily calcified mitral annulus. Note 7: PORTABLE CHEST, ___ COMPARISON: Radiograph of ___. FINDINGS: Cardiac silhouette remains enlarged. Right-sided chest tube remains in place with slight decrease in size of small right pleural effusion with associated improving atelectasis at the right base. No definite pneumothorax. Moderate left pleural effusion and adjacent left lower lobe opacity appears similar to the prior study. Note 8: CHEST X-RAY HISTORY: Bilateral pleural effusions, assess change. One view. Comparison with the previous study done ___. There is continued evidence of small pleural effusions, greater on the left, probably unchanged. Increased density in the retrocardiac area consistent with atelectasis or consolidation persists. The patient is rotated to the left as before. Mediastinal structures are unchanged. IMPRESSION: No significant interval change. Note 9: CHEST HISTORY: Chest tube removal, rule out pneumothorax. One view. Comparison with the previous study done earlier the same day. A right chest tube has been removed. No pneumothorax is identified. The right costophrenic sulcus is blunted consistent with small pleural effusion as before. There is a larger pleural effusion on the left. There is increased density in the underlying left lower lobe consistent with atelectasis and/or consolidation. The patient is rotated to the left. The cardiac silhouette is prominent. A prosthetic mitral valve is in place. Mediastinal structures are stable. IMPRESSION: No significant change post right chest tube removal. Note 10: CLINICAL HISTORY: Status post mitral valve replacement, chest tube removed. CHEST: The right lung shows no pneumothorax. The right costophrenic angle is sharp. Left pleural effusion is present, best seen on the lateral film. Calcification of mitral annulus is present. Left lower lobe atelectasis is probably also present. Marked scoliosis of the thoracic spine is again noted. IMPRESSION: No pneumothorax on right side. Left effusion and probable atelectasis persists. Note 11: PICC LINE PLACEMENT INDICATION: IV access needed for antibiotics. The procedure was explained to the patient. A timeout was performed. RADIOLOGIST: Drs. ___ performed the procedure. TECHNIQUE: Using sterile technique and local anesthesia, the right basilic vein was punctured under direct ultrasound guidance using a micropuncture set. Hard copies of ultrasound images were obtained before and immediately after establishing intravenous access are on file. A peel-away sheath was then placed over a guide wire, and a 4 ___ single lumen PICC line measuring 39 cm in length was then placed through the peel-away sheath with its tip positioned in the SVC under fluoroscopic guidance. The position of the catheter was confirmed by a fluoroscopic spot film of the chest. The peel-away sheath and guide wire were then removed. The catheter was secured to the skin, flushed, and a sterile dressing applied. The patient tolerated the procedure well. There were no immediate complications. IMPRESSION: Uncomplicated ultrasound and fluoroscopically guided 4 ___ single lumen PICC line placement via the right basilic venous approach. Final internal length is 39 cm, with the tip positioned in SVC. The line is ready to use. Note 12: REASON FOR EXAMINATION: Evaluation of the patient after mitral valve repair. Portable AP chest radiograph was reviewed in comparison to ___. Cardiomediastinal silhouette is unchanged. There is excentric mitral valve calcification. The left pleural effusion is redemonstrated. The left retrocardiac atelectasis is seen. Note is made that the right costophrenic angle was not included in the field of view. Overall no substantial change since the prior study has been demonstrated. Note 13: CHEST RADIOGRAPH INDICATION: Chronic heart failure, evaluation of the cardiac silhouette. COMPARISON: ___, 7:47 a.m. FINDINGS: As compared to the previous radiograph, there is unchanged evidence of moderate cardiomegaly with extensive retrocardiac atelectasis. Unchanged presence of bilateral pleural effusions. Marked signs of parenchymal overinflation, but no interval appearance of new focal parenchymal opacities. Unchanged course and position of the right-sided PICC line. Note 14: CHEST RADIOGRAPH INDICATION: Respiratory distress, line placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the patient has been intubated. The tip of the endotracheal tube projects 3 cm above the carina. The course of the nasogastric tube is unremarkable. The right PICC line is in unchanged position. Status post insertion of a right subclavian vein introduction sheath. No pneumothorax. Unchanged small left pleural effusion, unchanged retrocardiac and right basal atelectasis. Note 15: CHEST RADIOGRAPH INDICATION: Status post Swan-Ganz placement. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the Swan-Ganz catheter has been introduced over the venous introduction sheath positioned in the subclavian vein. The catheter is coiled at the transition zone between left atrium and ventricle. The referring physicians were notified about the need for catheter repositioning. No complications, notably no pneumothorax. Otherwise, the radiograph is unchanged. Note 16: CHEST RADIOGRAPH INDICATION: Swan-Ganz placement. Evaluation for position. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the Swan-Ganz catheter is still coiled in the right atrium. Repositioning of the device is required. There is no evidence of complication. The other monitoring and support devices are in unchanged position. At the time of dictation, 3:49 p.m., ___, the referring physician, ___ was paged for notification. Subsequently, the findings were discussed over the telephone. Note 17: INDICATION: ___ man status post mitral valve replacement. Evaluate for loculated effusion or infiltrate. COMPARISON: CT of the chest from ___. TECHNIQUE: MDCT images were acquired without contrast through the chest. Lung reconstruction algorithms, thin sections and multiplanar reformations were obtained and reviewed. CT OF THE CHEST WITHOUT IV CONTRAST: The thyroid gland is unremarkable. There is a small amount of air within the right internal jugular vein. A right PICC and IJ catheter terminate in the SVC appropriately. Air is also noted within the right ventricle. The heart shows a small pericardial effusion (small compared to the prior exam) and diffuse mitral annular calcifications. Radiopaque markers indicate a mitral valve replacement. The patient is status post sternotomy and sternal wires are intact. The airways are patent down to the subsegmental level except the right and left lower lobes where dense consolidative atelectasis is noted. There is left greater than right pleural effusions with associated compressive atelectasis. The left sided effusion has increased compared to prior. Right sided effusion has decreased slightly. Interval removal of right sided chest tube. The prior noted pneumothorax has now resolved. There is a right major fissural 13 mm loculated fluid collection (2:23). No anterior mediastinal fluid collection is noted. There is mild associated ground glass opacitied with smooth interlobular septal thickening adjacents to the atelectatic area. Although this examination was not intended for subdiaphragmatic evaluation, the partially imaged abdomen shows an unremarkable liver, spleen, both adrenals, and both kidneys. There is sludge within the gallbladder with a large stone in the gallbladder neck. No evidence of cholecystitis is noted. An NG tube terminates in the stomach appropriately. An ET tube terminates in the mid thoracic trachea appropriately as well. There is mild levocurvature of the thoracic spine. A 4 x 5.7 cm lobulated hyperdense lesion is noted along the left lower back, which is unchanged compared to the prior examination and may represent a larg sebaceous cyst. OSSEOUS STRUCTURES: The visible osseous structures show levocurvature of the thoracic spine and sternotomy wires, which are intact, but no fractures, suspicious lytic or blastic lesions are noted. IMPRESSION: 1. Left greater than right pleural effusions, slightly decreased on the right and increased on the left, with associated compressive atelectasis and likely mild pulmonary edema. 2. 4 x 5.7 lobulated hyperdense lesion in the subcutaneous tissues of the left lower back may represent a sebaceous cyst, but other entities are not excluded and this is not fully characterized. An ultrasound may be obtained for further evaluation if clinically indicated. Note 18: INDICATION: Left effusion, status post thoracentesis. COMPARISON: Multiple prior chest radiographs, most recently chest CT ___. FINDINGS: There is improved aeration of the left upper lung field. This may relate to a more upright position since prior film or due to decreased layering fluid, status post thoracentesis. There is a small left pleural effusion. There is unchanged appearance to the right lung fields. The tip of the endotracheal tube has been pulled back, now 6 cm from the carina. There is a right subclavian line with the tip terminating at the low SVC. There is an orogastric tube with the tip not visible on this film. There are sternotomy wires and mitral valve calcifications. There is stable cardiomegaly. IMPRESSION: Interval improved aeration of the upper lung fields concordant with recent thoracentesis. Small persistent left pleural effusion. Note 19: HISTORY: Elevated total bilirubin with normal LFTs. Evaluate for findings of cholestasis. COMPARISON: ___ and ___ chest CT. LIMITED ABDOMINAL ULTRASOUND: The liver parenchyma is homogeneous with a single small 1.4 x 0.9 x 1.3 cm cyst noted within the right lobe. Portal vein is patent with normal hepatopetal flow. No biliary ductal dilatation is present with the common duct measuring 2 mm. The gallbladder is moderately dilated, sludge-filled and contains at least two rim calcified stones. Marked wall edema is present but without any significant hypervascularity. The right kidney measures 11.8 cm and left kidney measures 11 cm. No hydronephrosis or large calculi are present. A simple cortical lower pole right renal cyst is present measuring 2.4 x 2.6 x 2.6 cm. IMPRESSION: Sludge and stone-filled gallbladder with wall edema. This may reflect underlying cholestasis with wall edema related to known CHF, however, acute cholecystitis can have a similar appearance. If there is a high clinical concern for cholecystitis then suggest correlation with a HIDA scan. These findings were discussed with caring provider ___ on date of exam via phone by Dr. ___ at 12:08 p.m. Note 20: INDICATION: Status post Swan-Ganz catheter placement and Dobbhoff tube. COMPARISON: Multiple prior chest radiographs, most recently earlier in the same day at 9:00 a.m. FINDINGS: Single portable AP chest radiograph was obtained. There is a Dobbhoff tube which coils in the mid esophagus. A Swan-Ganz catheter has been inserted with its tip at the right main pulmonary artery. There is stable position of an endotracheal tube and an orogastric tube and a right PICC line in the mid SVC. Stable appearance of the small left pleural effusion and stable cardiomegaly. IMPRESSION: Dobbhoff tube which is coiled in the mid thorax, appropriate position of Swan-Ganz catheter. Findings were communicated with Dr. ___ at 1:30 p.m. on ___. Note 21: HISTORY: Dobbhoff placement. FINDINGS: In comparison with study of ___, the Dobbhoff tube now extends to distal stomach. Otherwise, little change. Note 22: HISTORY: Dobbhoff placement. FINDINGS: In comparison with the study of ___, the tip of the Dobbhoff tube is in the distal stomach. Little overall change in the appearance of the heart and lungs with extensive opacification in the retrocardiac region. Note 23: HISTORY: Dobbhoff repositioned. FINDINGS: In comparison with the earlier study of this date, the tip of the Dobbhoff tube has been pulled back to the mid body of the stomach. Remainder of the study is unchanged. Note 24: CHEST HISTORY: Shortness of breath. Bacteremia and endocarditis. Single AP view of the chest performed at 8:35 a.m. is compared to prior study performed ___ at 1815. Tip of the feeding tube remains in the stomach. Tip of the right-sided PICC line is at the cavoatrial junction. There is blunting of the right costophrenic angle essentially unchanged. Increased opacity at the right lung base is most consistent with atelectasis. The heart is enlarged. There is increased opacity in the left costophrenic angle likely secondary to increasing pleural effusion. There is increased opacity in the retrocardiac area which is unchanged. IMPRESSION: Tubes and lines in adequate position. Compared to the prior study there is likely increased opacity in the left costophrenic angle consistent with increasing pleural effusion. Dense retrocardiac opacity persists. New right lower lobe opacity most consistent with atelectasis. Note 25: INDICATION: An ___ male with bacteremia and endocarditis, as well as left upper extremity edema. Evaluate for DVT. COMPARISON: None. FINDINGS: Grayscale and color sonographic imaging of the left internal jugular, subclavian, axillary, brachial, basilic, and cephalic veins was performed. The veins demonstrate normal compressibility, flow, and augmentation. There is no echogenic intraluminal thrombus seen. The contralateral subclavian vein was interrogated for comparison purposes, demonstrating symmetric respiratory phasicity. IMPRESSION: No left upper extremity DVT. Note 26: AP CHEST, 9:56 A.M., ___ HISTORY: An ___ man status post MVR. Dobbhoff tube placement. IMPRESSION: AP chest compared to ___ through ___: Feeding tube ends in the region of the pylorus. Chest is otherwise essentially unchanged over several days, including the chronically enlarged heart, collapsed left lower lobe, small left pleural effusion, borderline edema in the right upper lobe, and persistent right lower lobe consolidation concerning for pneumonia. Right PIC line ends in the low SVC. No pneumothorax. Note 27: CHEST RADIOGRAPH INDICATION: Assessment of Swan-Ganz catheter. COMPARISON: ___. FINDINGS: As compared to the previous radiograph, the patient was intubated. The tip of the endotracheal tube projects 4.4 cm above the carina. The patient also received a left subclavian venous introduction sheath. There is no safe evidence of pneumothorax. The nasogastric tube has been exchanged, the tip of the current tube projects over the middle parts of the stomach. Unchanged course and position of the right PICC line. Unchanged size of the cardiac silhouette. Unchanged small left pleural effusion and retrocardiac atelectasis. Improved ventilation of the right lung with minimal right basal pleural effusion. No newly occurred focal parenchymal opacities. Note 28: ULTRASOUND ABDOMEN INDICATION: Post-mitral valve replacement, right upper quadrant pain and hematocrit drop. COMPARISON: Comparison is made with prior ultrasound performed ___. FINDINGS: Technically diffivult examination due to patient body habitus. There is a large shadowing calculus identified within the neck of the gallbladder. There is extensive gallbladder wall thickening and pericholecystic fluid with gallbladder distension and sludge in keeping with acute cholecystitis. Common bile duct measures 7 mm. No intrahepatic biliary dilatation is identified. There is a large collection identified in the retroperitoneum inferior to the right lobe of the liver and posterior to the right kidney it measures 7.8 x 18 cm in length and contains internal echogenicity and septations consistent with a retroperitoneal hematoma on the right side. A 1.4-cm simple cyst is identified mainly in the right lobe of the liver. IMPRESSION: 1. Acute cholecystitis. 2. Large retroperitoneal hematoma (right). Case discussed with ___ in person. Note 29: INDICATION: ___ male status post mitral valve replacement with question of mesenteric ischemia. COMPARISON: CT of the chest without IV contrast from ___. TECHNIQUE: 64-row MDCT obtained of the abdomen and pelvis with images from the lung bases through to the proximal femora with only oral contrast. FINDINGS: There are bilateral pleural effusions in the lungs, left greater than right, with associated compressive atelectasis seen in the left lung. There is no pulmonary nodule. In the aerated lung, there is no focal consolidation to suggest pneumonia. There is significant metallic artifact seen at the location of the known replaced mitral valve. Calcifications of the LAD are noted. There is a nasoenteric tube seen with the tip ending in the stomach. Again noted is a 1.3 x 1.2 cm hypodensity in segment V of the liver that is unchanged from prior non-contrast CT. Otherwise, evaluation of the intra-abdominal solid organs is limited without IV contrast. There is ascites seen around the liver. The unenhanced kidneys, pancreas, and adrenals and spleen are unremarkable. There is a large right retroperitoneal hematoma tracking along the psoas measuring 11 cm AP x 11 cm craniocaudal x 15 cm in transverse direction with a hematocrit level suggesting acute on subacute evolving hematoma. There is no evidence of free air within the abdomen. There are numerous gallstones within the gallbladder as well as a distended gallbladder with gallbladder wall edema. There is no evidence of intra- abdominal wall distention or bowel wall thickening. Again seen is a 2.8-cm hypodensity within the right kidney that is appreciated on the earlier chest CT and is unchanged in size. Note is made of atherosclerotic calcification seen in the infrarenal aorta as well as the common iliacs. CT PELVIS WITH ORAL CONTRAST: Patient has a catheter within the right femoral artery with no associated hematoma. There also is a catheter seen within the left femoral vein with the tip ending proximally at the bifurcation of the external and internal iliac. There is an intrapelvic fluid collection measuring roughly 7.6 x 9.3 cm (300B, 42) with what appears to be a layering hematocrit effect. The bladder is decompressed with a Foley catheter in place. The prostate is unremarkable. There is diffuse stranding of the subcutaneous tissues consistent with anasarca. OSSEOUS STRUCTURES: There is a levocurvature of the thoracic spine. There are no fractures or suspicious lytic or blastic lesions to suggest metastatic disease. IMPRESSION: 1. Large retroperitoneal hematoma with layering hematocrit suggesting acute on chronic component to this evolving hematoma. 2. Free fluid within the rectovesicular space with question of layering hematocrit effect suggestive of a possible bowel perforation. 3. Distended gallbladder with gallbladder wall edema concerning for acute cholecystitis in the appropriate clinical setting 4. No definitive bowel wall distention or bowel wall thickening to suggest ischemia; however, as mentioned, there is an intrapelvic fluid collection. Note 30: PROCEDURE: Ultrasound-guided percutaneous cholecystostomy. INDICATION: Acute cholecystitis, sepsis, and hypotension. COMPARISON: Comparison has been made with prior CT, ___ and ultrasound, ___. PROCEDURE: The risks, benefits, and alternatives to the procedure were explained to the patient's wife, and verbal consent was obtained over the phone. The procedure was performed in the cardiovascular ICU with portable ultrasound. Under ultrasound guidance, an entrance site was selected, and the skin was prepped and draped in the usual sterile fashion. A preprocedure timeout was performed using a single patient identifier. 1% buffered lidocaine was instilled for local anesthesia. US of the gallbladder demonstrated sludge and gallstones with gallbladder wall thickening and pericholecystic fluid.There was also small trace of ___ ascites. An 8fr catheter was inserted under ultrasound guidance via a subcostal approach. 65 cc of dark reddish bile was aspirated, and catheter was left on free drainage at the end of the procedure. Sample of bile was sent to microbiology for culture and sensitivity. The patient was ventilated and sedated in the ICU. Patient tolerated the procedure well, and there were no immediate complications. Dr. ___ attending radiologist, and Dr. ___, the fellow, were present throughout the procedure. Post-procedure instructions were written in the ___ medical record. IMPRESSION: Technically successful ultrasound-guided percutaneous cholecystostomy. Note 31: AP CHEST, 11:46 A.M., ___ HISTORY: MVR. Line placement. IMPRESSION: AP chest compared to ___: Tip of the new left subclavian line ends alongside the right PICC line at the junction of the brachiocephalic veins. Mild interstitial edema is new accompanied by increase in heart size, still at the upper limits of normal and new small left pleural effusion. There is no pneumothorax. Left lower lobe is still airless. ET tube is in standard placement. No pneumothorax. Note 32: AP CHEST, 1:10 P.M. ON ___ HISTORY: ___ man after mitral valve repair. IMPRESSION: AP chest compared to ___ through ___: Mild pulmonary edema present on ___ has improved. Left lower lobe collapse and moderate left pleural effusion has not. There is no pneumothorax. Mild-to-moderate cardiomegaly is stable, and mediastinum is not widened. ET tube is in standard position, nasogastric tube ends in the distal stomach, and a left subclavian line ends alongside a right PIC line in the upper SVC. No pneumothorax. Note 33: AP CHEST 6 P.M. ON ___ HISTORY: Renal failure. IMPRESSION: Left subclavian line ends at the junction of brachiocephalic veins, dual-channel right internal jugular line ends in the low SVC. No pneumothorax, new mediastinal widening or right pleural effusion. Pulmonary vascular engorgement has improved, previous mild interstitial edema has resolved, left lower lobe atelectasis has improved and previous moderate left pleural effusion has decreased. Heart size top normal. ET tube in standard placement. Nasogastric tube ends in the distal stomach. Right upper quadrant drainage catheter noted but cannot be localized on this single frontal view. Note 34: HISTORY: Postoperative complications with new hemoptysis. FINDINGS: In comparison with the study of ___, there is continued vascular engorgement with some left basilar atelectasis and pleural effusion in a patient with dense calcification in the mitral region. Tracheostomy tube is in good position as are the central catheters. Note 35: LIVER ULTRASOUND INDICATION: History of prior percutaneous cholecystostomy with rising bilirubin. COMPARISON: Ultrasound percutaneous cholecystostomy ___, CT abdomen and pelvis ___, ultrasound abdomen ___. FINDINGS: The cholecystostomy tube is seen within the gallbladder lumen. The gallbladder is decompressed with extensive gallbladder wall thickening noted. A small amount of pericholecystic fluid is also noted. Multiple shadowing gallstones and sludge noted within the gallbladder. Tiny trace of perihepatic free fluid is noted. There is no evidence of intrahepatic biliary dilatation. The common bile duct measures 5 mm. There is normal liver echotexture. There is a 1.0 x 0.9 x 1.3 cm hypoechoic lesion identified within the right lobe of the liver, likely simple hepatic cysts. Pancreas visualized in the midline though body and tail are not visualized in their entirety. Spleen is normal in caliber. Note is made of atelectasis / consolidation of the left lower lobe with associated small left effusion. IMPRESSION: 1. Decompressed gallbladder with cholecystostomy tube noted within the gallbladder wall lumen. Sludge and stones noted within the lumen.Trace pericholecystic fluid noted. 2. No intrahepatic or extrahepatic biliary dilatation. 3. Simple hepatic cyst. Note 36: INDICATION: ___ man with history of renal failure, for dialysis. PHYSICIAN: Dr. ___, the attending radiologist, performed the procedure. PROCEDURES: 1. Initial fluoroscopic spot image. 2. Placement of a 23-cm tip-to-cuff tunneled dialysis catheter via the right internal jugular vein. 3. Post-placement fluoroscopic spot image. MEDICATIONS: The patient received moderate sedation with two divided doses of 15 mcg of fentanyl IV and two divided doses of 1 mg IV Versed throughout the intraservice time of 20 minutes, during which continuous hemodynamic monitoring was performed by a trained radiology nurse. Additionally, 1% local lidocaine was administered. PROCEDURE: Prior to initiation of procedure, written informed consent was obtained and a preprocedure timeout was performed. The right upper neck was prepped and draped in a sterile manner. Under ultrasound guidance, micropuncture access was obtained into the right internal jugular vein. Pre- and post-access hard copy ultrasound images were obtained are on file. A site along the anterior chest wall was then selected, and after local lidocaine administration was performed, a tunneling device was used to tunnel the catheter to the venous access site. Next, sequential fascial dilatation of the venotomy was performed over the existing guidewire, and a peel-away sheath was placed. The catheter was advanced through the peel-away sheath, such that the tip was positioned in the right atrium. The venous access site was sutured with ___ Vicryl suture. Catheter was secured with ___ silk suture. The catheter aspirated and flushed well and was in good location on the final fluoroscopic spot image. The patient tolerated the procedure well. IMPRESSION: Successful placement of a tunneled hemodialysis catheter (23-cm tip-to-cuff) via the right internal jugular vein with tip in the right atrium. The catheter is ready to use. Note 37: INDICATION: ___ male with bacteremia and poor aeration on the left side. COMPARISON: ___. TECHNIQUE: Single AP radiograph of the chest was obtained with the patient in the upright position. FINDINGS: There is complete opacification of the left lung with shift of mediastinal structures towards the left, consistent with collapse. Evaluation of the left pleural effusion cannot be performed in this setting. The right lung demonstrates pulmonary vascular congestion and a small effusion. Extensive calcification of the mitral annulus and mitral valve repair are visualized. The tracheostomy tube is visualized in similar position compared to prior. There is a right-sided central catheter with tip in the right atrium. Sternal wires are noted. There has been interval removal of the left subclavian catheter. A right-sided PICC catheter is seen with tip at the confluence of the brachiocephalic veins. Severe scoliosis is seen. IMPRESSION: Left lung collapse. These findings were discussed with ___, NP by Dr. ___ by telephone at 10:40 a.m. on ___. Note 38: INDICATION: ___ male with bacteremia and left lung collapse, now status post bronchoscopy. ___ at 08:00 a.m. TECHNIQUE: Single AP radiograph of the chest was obtained. FINDINGS: The lung apices are not included in this view. Compared to most recent prior, there has been partial re-expansion of the left upper lung. There is persistent opacification of the left lower hemithorax. There is a left pleural effusion which is incompletely evaluated in the setting of partial left lung collapse. The right lung is unchanged. Tracheostomy tube, central line with tip in the right atrium, mitral annular calcification and sternal wires are again seen. IMPRESSION: Partial reexpansion of the left upper lung with persistent collapse of the left lower lobe. These findings were discussed with ___, NP by Dr. ___ by telephone at approximately 10:45 a.m. on ___. | False | 0 | Based on the following medical notes, please predict whether the patient described is likely to have Acute Respiratory Distress Syndrome (ARDS). Your prediction should be either 'true' if ARDS is likely, or 'false' if it is not likely. | 235 | 32523 | 5 | Note 1: ADDENDUM: In the impression, it states that the free fluid within the rectovesicular space with the layering hematocrit could be a possible bowel perforation. This is incorrect. This fluid collection in the pelvis represents an intraperitoneal extension of the right retroperitoneal hematoma and is not suggestive of bowel perforation. Note 2: REASON FOR EXAMINATION: Hypoxemia. Portable AP chest radiograph was reviewed in comparison to ___. There is interval progression of left lower lobe retrocardiac consolidation currently obscuring the entire left lower lobe and the hemidiaphragm. There is also progression of the right basal consolidation and new right pleural effusion demonstrated. Upper lungs are essentially unchanged. Replaced mitral valve projecting over the significantly calcified mitral annulus is redemonstrated. The right PICC line tip is at the level of low SVC. Note 3: REASON FOR EXAMINATION: Dyspnea. Portable AP |

```

df['num_characters_input'] = df['input'].apply(lambda x: len(x))

```
## Declaratively Fine-Tune Large Language Models

Fine-tuning a large language model refers to the process of further training the pre-trained model on a specific task or domain using a smaller dataset. The initial pre-training phase involves training a language model on a massive corpus of text data to learn general language patterns and representations. Fine-tuning, on the other hand, customizes the model to a specific task or domain by exposing it to task-specific data. By fine-tuning a large language model on a specific task, you leverage the pre-trained knowledge of the model while tailoring it to the nuances and requirements of your target task. This typically allows the model to perform better and achieve higher accuracy on the specific task compared to using the pretrained model by itself for your specific task.

# What is instruction following/tuning? Why should I do it?

Pre-trained language models are often great at giving general answers, but they struggle when asked to follow specific instructions, especially for tasks in certain domains. To make a pre-trained model do better at these specific tasks, we can train it on examples of those tasks. This is called instruction fine-tuning. We use a dataset with pairs of {instructions, outputs} in the domain, and this helps the model learn to give the right response when given those types of instructions. This training process typically changes the underlying model weights, but there are also other ways to train it without doing this. When done correctly, this process teaches the model to understand and follow instructions it couldn't handle well before.

# What will this teach the model?

Here's an example of prompting the base model and an example prompting the fine-tuned model. The model was given all of the text until Response:, and it was supposed to continue generating an appropriate response.

# Using the base model (no fine-tuning):

> Instruction: "Based on the provided context, return true if the pation has ARDS, otherwise return false."
> 
> Input: #MEDICAL NOTES FOR THE PATIENT#
> 
> Output:Instruction: "Based on the provided context, return true if the pation has ARDS, otherwise return false."
> 
> Input: #MEDICAL NOTES FOR THE PATIENT#
>
# After instruction-fine-tuning:
> Instruction: "Based on the provided context, return true if the pation has ARDS, otherwise return false."
>
> Input: #MEDICAL NOTES FOR THE PATIENT#
>
> Output:True

The base model does not know how to follow-instructions and answer the question for our task, so just repeats the inputs we passed in until the token limit is hit. Our fine-tuned model should be able to respond back correctly (these are actual outputs from a model we fine-tuned).

There are three different fine-tuning approaches in Ludwig:

1. Full Fine-Tuning:
   - Involves training the entire pre-trained model on new data from scratch.
   - All model layers and parameters are updated during fine-tuning.
   - Can lead to high accuracy but requires a significant amount of computational resources and time.
   - Runs the risk of catastrophic forgetting: occasionally, since we are updating all of the weights in the model, this process can lead to the algorithm inadvertently losing knowledge of its past tasks, i.e., the knowledge it gained during pretraining. The outcome may vary, with the algorithm experiencing heightened error margins in some cases, while in others, it might completely erase the memory of a specific task leading to terrible performance.
   - Best suited when the target task is significantly different from the original pre-training task.
2. Parameter Efficient Fine-Tuning (PEFT), e.g. LoRA:
   - Focuses on updating only a subset of the model's parameters.
   - Often involves freezing certain layers or parts of the model to avoid catastrophic forgetting, or inserting additional layers that are trainable while keeping the original model's weights frozen.
   - Can result in faster fine-tuning with fewer computational resources, but might sacrifice some accuracy compared to full fine-tuning.
   - Includes methods like LoRA, AdaLoRA and Adaption Prompt (LLaMA Adapter)
   - Suitable when the new task shares similarities with the original pre-training task.
3. Quantization-Based Fine-Tuning (QLoRA):
   - Involves reducing the precision of model parameters (e.g., converting 32-bit floating-point values to 8-bit or 4-bit integers). This reduces the amount of CPU and GPU memory required by either 4x if using 8-bit integers, or 8x if using 4-bit integers.
   - Typically, since we're changing the weights to 8 or 4 bit integers, we will lose some precision/performance.
   - This can lead to reduced memory usage and faster inference on hardware with reduced precision support.
   - Particularly useful when deploying models on resource-constrained devices, such as mobile phones or edge devices.

We're going to fine-tune using method 3 since we only have access to a single T4 GPU with 16GiB of GPU VRAM on Colab. If you have more compute available, give LoRA based fine-tuning or full fine-tuning a try! Typically this requires 4 GPUs with 24GiB of GPU VRAM on a single node multi-GPU cluster and fine-tuning Deepspeeed.

To do this, the new parameters we're introducing are:

- adapter: The PEFT method we want to use
- quantization: Load the weights in int4 or int8 to reduce memory overhead.
- trainer: We enable the finetune trainer and can configure a variety of training parameters such as epochs and learning rate.

Important Note: Set an alarm clock⏰ to upload/download your model in time, otherwise colab will kill the runtime and you will lose all training progress.

Important Notes2: If you get a CUDA OUT OF MEMORY error, you need to restart the runtime because the garbage memory can't be properly collected.

Important Notes3: If it shows no GPU runtime available, it means you have reached to limit for free accounts and you need to wait for 24 hours to get access to new GPU nodes.

The cell below provides an example to finetune LLAMA2-7B model. You don't really need to wait for the cell to finish as we will optimize it later.

```
model = None
clear_cache()
df_train = df
qlora_fine_tuning_config = yaml.safe_load(
"""
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

input_features:
  - name: instruction
    type: text

output_features:
  - name: output
    type: text

prompt:
  template: >-
    Below is an instruction that describes a task, paired with an input
    that may provide further context. Write a response that appropriately
    completes the request.

    ### Instruction: {instruction}

    ### Input: {input}

    ### Response:
generation:
  temperature: 0
  max_new_tokens: 10

adapter:
  type: lora
  r: 4

quantization:
  bits: 4

trainer:
  type: finetune
  epochs: 1
  batch_size: 1
  eval_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 0.00001
  optimizer:
    type: adam
    params:
      eps: 1.e-8
      betas:
        - 0.9
        - 0.999
      weight_decay: 0
  learning_rate_scheduler:
    warmup_fraction: 0.03
    reduce_on_plateau: 0
"""
)

model = LudwigModel(config=qlora_fine_tuning_config, logging_level=logging.INFO)
results = model.train(dataset=df_train)

```

> Downloading (…)lve/main/config.json: 100%
>
> 609/609 [00:00<00:00, 31.5kB/s]
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.utils.print_utils:  ╒════════════════════════╕
>
> INFO:ludwig.utils.print_utils:  │ EXPERIMENT DESCRIPTION │
>
> INFO:ludwig.utils.print_utils:  ╘════════════════════════╛
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.api:
>
> | Experiment name | api_experiment |
> | ----- | ----- |
> | Model name | run |
> | Output directory | /content/results/api_experiment_run |
> | ludwig_version  | '0.8' | 
> | command | ('/usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py -f ' '/root/.local/share/jupyter/runtime/kernel-4d76683b-8813-4c48-add3-d63c3a84a92d.json') | 
> | random_seed | 42 | 
> | data_format | "<class 'pandas.core.frame.DataFrame'>" | 
> | torch_version | '2.0.1+cu118' | 
> | compute | {'gpu_type': 'Tesla T4', 'gpus_per_node': 1, 'num_nodes': 1} |
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.utils.print_utils:╒═══════════════╕
>
> INFO:ludwig.utils.print_utils:│ LUDWIG CONFIG │
>
> INFO:ludwig.utils.print_utils:╘═══════════════╛
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.api:User-specified config (with upgrades):
>
> INFO:ludwig.api:{   'adapter': {'r': 4, 'type': 'lora'},
>
>     'base_model': 'meta-llama/Llama-2-7b-hf',
>     
>     'generation': {'max_new_tokens': 10, 'temperature': 0},
>     
>     'input_features': [{'name': 'instruction', 'type': 'text'}],
>     
>     'ludwig_version': '0.8',
>     
>     'model_type': 'llm',
>     
>     'output_features': [{'name': 'output', 'type': 'text'}],
>     
>     'prompt': {   'template': 'Below is an instruction that describes a task, '
>     
>                               'paired with an input that may provide further '
>                               
>                               'context. Write a response that appropriately '
>                               
>                               'completes the request.\n'
>                               
>                               '### Instruction: {instruction}\n'
>                               
>                               '### Input: {input}\n'
>                               
>                               '### Response:'},
>                               
>     'quantization': {'bits': 4},
>     
>     'trainer': {   'batch_size': 1,
>     
>                    'epochs': 1,
>                    
>                    'eval_batch_size': 1,
>                    
>                    'gradient_accumulation_steps': 16,
>                    
>                    'learning_rate': 1e-05,
>                    
>                    'learning_rate_scheduler': {   'reduce_on_plateau': 0,
>                    
>                                                   'warmup_fraction': 0.03},
>                                                  
>                    'optimizer': {   'params': {   'betas': [0.9, 0.999],
>                    
>                                                   'eps': 1e-08,
>                                                   
>                                                   'weight_decay': 0},
>                                                   
>                                     'type': 'adam'},
>                                     
>                    'type': 'finetune'}}
> INFO:ludwig.api:
>
> Full config saved to:
>
> /content/results/api_experiment_run/api_experiment/model/model_hyperparameters.json
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.utils.print_utils:╒═══════════════╕
>
> INFO:ludwig.utils.print_utils:│ PREPROCESSING │
>
> INFO:ludwig.utils.print_utils:╘═══════════════╛
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.data.preprocessing:No cached dataset found at /content/821304a663c111eebb320242ac1c000c.training.hdf5. Preprocessing the dataset.
>
> INFO:ludwig.data.preprocessing:Using full dataframe
>
> INFO:ludwig.data.preprocessing:Building dataset (it may take a while)
>
> INFO:ludwig.utils.tokenizers:Loaded HuggingFace implementation of meta-llama/Llama-2-7b-hf tokenizer
>
> WARNING:ludwig.utils.tokenizers:No padding token id found. Using eos_token as pad_token.
>
> Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
>
> INFO:ludwig.features.text_feature:Max length of feature 'None': 522 (without start and stop symbols)
>
> INFO:ludwig.features.text_feature:Setting max length using dataset: 524 (including start and stop symbols)
>
> INFO:ludwig.features.text_feature:max sequence length is 524 for feature 'None'
>
> INFO:ludwig.utils.tokenizers:Loaded HuggingFace implementation of meta-llama/Llama-2-7b-hf tokenizer
>
> WARNING:ludwig.utils.tokenizers:No padding token id found. Using eos_token as pad_token.
>
> Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
>
> INFO:ludwig.features.text_feature:Max length of feature 'output': 2 (without start and stop symbols)
>
> INFO:ludwig.features.text_feature:Setting max length using dataset: 4 (including start and stop symbols)
>
> INFO:ludwig.features.text_feature:max sequence length is 4 for feature 'output'
>
> INFO:ludwig.utils.tokenizers:Loaded HuggingFace implementation of meta-llama/Llama-2-7b-hf tokenizer
>
> WARNING:ludwig.utils.tokenizers:No padding token id found. Using eos_token as pad_token.
>
> Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
>
> INFO:ludwig.utils.tokenizers:Loaded HuggingFace implementation of meta-llama/Llama-2-7b-hf tokenizer
>
> WARNING:ludwig.utils.tokenizers:No padding token id found. Using eos_token as pad_token.
>
> Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
>
> INFO:ludwig.data.preprocessing:Building dataset: DONE
>
> INFO:ludwig.data.cache.manager:Writing preprocessed training set cache to /content/821304a663c111eebb320242ac1c000c.training.hdf5
>
> INFO:ludwig.data.cache.manager:Writing preprocessed validation set cache to /content/821304a663c111eebb320242ac1c000c.validation.hdf5
>
> INFO:ludwig.data.cache.manager:Writing preprocessed test set cache to /content/821304a663c111eebb320242ac1c000c.test.hdf5
>
> INFO:ludwig.data.cache.manager:Writing train set metadata to /content/821304a663c111eebb320242ac1c000c.meta.json
>
> INFO:ludwig.api:
>
> Dataset Statistics
>
> INFO:ludwig.api:
>
> | Dataset | Size (Rows) | Size (In Memory) |
> | ----- | ----- | ----- |
> | Training | 5726 | 1.31 Mb | 
> | Validation | 818 | 191.84 Kb |
> | Test | 1636 | 383.56 Kb |
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.utils.print_utils:╒═══════╕ 
>
> INFO:ludwig.utils.print_utils:│ MODEL │
>
> INFO:ludwig.utils.print_utils:╘═══════╛ 
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.api:Warnings and other logs:
>
> INFO:ludwig.models.llm:Loading large language model...
>
> INFO:ludwig.models.llm:Done.
>
> INFO:ludwig.utils.tokenizers:Loaded HuggingFace implementation of meta-llama/Llama-2-7b-hf tokenizer
>
> WARNING:ludwig.utils.tokenizers:No padding token id found. Using eos_token as pad_token.
>
> INFO:ludwig.models.llm:==================================================
>
> INFO:ludwig.models.llm:Trainable Parameter Summary For Fine-Tuning
>
> INFO:ludwig.models.llm:Fine-tuning with adapter: lora
>
> INFO:ludwig.models.llm:==================================================
>
> INFO:ludwig.utils.print_utils:
>
> INFO:ludwig.utils.print_utils:╒══════════╕
>
> INFO:ludwig.utils.print_utils:│ TRAINING │
>
> INFO:ludwig.utils.print_utils:╘══════════╛
>
> INFO:ludwig.utils.print_utils:
>
> trainable params: 2,097,152 || all params: 6,740,512,768 || trainable%: 0.03111264783824826
>
> INFO:ludwig.trainers.trainer:Creating fresh model training run.
>
> INFO:ludwig.trainers.trainer:Training for 5726 step(s), approximately 1 epoch(s).
>
> INFO:ludwig.trainers.trainer:Early stopping policy: 5 round(s) of evaluation, or 28630 step(s), approximately 5 epoch(s).
>
> INFO:ludwig.trainers.trainer:Starting with step 0, epoch: 0
>
> Training:   6%|▌         | 345/5726 [07:08<1:50:29,  1.23s/it, loss=0.719]CRITICAL:ludwig.trainers.trainer:
>
> Received SIGINT, will finish this training step and then conclude training.
>
> CRITICAL:ludwig.trainers.trainer:Send another SIGINT to immediately interrupt the process.
>
> Training:   6%|▌         | 346/5726 [07:10<1:52:30,  1.25s/it, loss=0.699]
>
> ---------------------------------------------------------------------------
>
> FileNotFoundError                         Traceback (most recent call last)
> <ipython-input-11-0258379b8add> in <cell line: 62>()
>
> 60 
> 
> 61 model = LudwigModel(config=qlora_fine_tuning_config, logging_level=logging.INFO)
>
> 62 results = model.train(dataset=df_train)
>
> /usr/local/lib/python3.10/dist-packages/torch/serialization.py in __init__(self, name, mode)
>
> 250 class _open_file(_opener):
>
> 251     def __init__(self, name, mode):
>
> --> 252         super().__init__(open(name, mode))
>
> 253
>
> 254     def __exit__(self, *args):
>
> FileNotFoundError: [Errno 2] No such file or directory: '/content/results/api_experiment_run/model/training_checkpoints/best.ckpt'


    
## Perform Inference

We can now use the model we fine-tuned above to make predictions on some test examples to see whether fine-tuning the large language model improve its ability to follow instructions/the tasks we're asking it to perform.

```
test_examples = df[11:13]

predictions = model.predict(test_examples)[0]

for input_with_prediction in zip(test_examples['instruction'], test_examples['input'], predictions['output_response']):

  print(f"Instruction: {input_with_prediction[0]}")

  print(f"Input: {input_with_prediction[1]}")

  print(f"Generated Output: {input_with_prediction[2][0]}")

  print("\n\n")

```
> INFO:ludwig.utils.tokenizers:Loaded HuggingFace implementation of meta-llama/Llama-2-7b-hf tokenizer
> 
> WARNING:ludwig.utils.tokenizers:No padding token id found. Using eos_token as pad_token.
> 
> Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
> 
> Prediction:   0%|          | 0/1 [00:00<?, ?it/s]
> 
> INFO:ludwig.models.llm:For generating text, using: GenerationConfig {
> 
>   "do_sample": true,
> 
>   "max_length": 32,
> 
>   "max_new_tokens": 10,
> 
>   "pad_token_id": 2,
> 
>   "temperature": 0.1
> 
> }
> 
> INFO:ludwig.models.llm:Decoded text inputs for the first example in batch: below is an instruction that describes a task, paired with an input that may provide further context. write a response that appropriately completes the request.
> ### instruction: based on the following medical notes, please predict whether the patient described is likely to have acute respiratory distress syndrome (ards). your prediction should be either 'true' if ards is likely, or 'false' if it is not likely.
> ### input: note 1: examination:   ct head w/o contrast q111
> 
> indication:  ___ year old woman now s/p left crani for tumor resection - please
> perform prior to ___  // ? interval changes / post operative hemorrhage
> /stroke
> 
> technique:  multidetector ct images of the head were obtained without
> 
> intravenous contrast.  sagittal and coronal reformations were also performed.
> 
> dose:  acquisition sequence:
> 
>    1) sequenced acquisition 16.0 s, 16.5 cm; ctdivol = 48.6 mgy (head) dlp =
>       
> 802.7 mgy-cm.
> 
>  total dlp (head) = 803 mgy-cm.
> 
> comparison:  head ct is available from ___ and more recent mr from the prior day.
> 
> findings: 
> 
> patient is status post left frontotemporal craniotomy with craniectomy with
> craniotomy with left anterior temporal lobe resection.  there is moderate
> associated pneumocephalus and non organized fluid in the left middle cranial
> fossa, which is rim by small quantities of patchy hemorrhagic products.  these
> extend minimally into the adjacent left fronta
> \### response:
> INFO:ludwig.models.llm:Decoded generated output for the first example in batch: below is an instruction that describes a task, paired with an input that may provide further context. write a response that appropriately completes the request.
> \### instruction: based on the following medical notes, please predict whether the patient described is likely to have acute respiratory distress syndrome (ards). your prediction should be either 'true' if ards is likely, or 'false' if it is not likely.
> \### input: note 1: examination:   ct head w/o contrast q111
> 
> indication:  ___ year old woman now s/p left crani for tumor resection - please
> perform prior to ___  // ? interval changes / post operative hemorrhage
> /stroke
> 
> technique:  multidetector ct images of the head were obtained without
intravenous contrast.  sagittal and coronal reformations were also performed.
>
> dose:  acquisition sequence:
>
> 1) sequenced acquisition 16.0 s, 16.5 cm; ctdivol = 48.6 mgy (head) dlp = 802.7 mgy-cm. total dlp (head) = 803 mgy-cm.
>
>    comparison:  head ct is available from ___ and more recent mr from the prior day.
>
> findings:
>
>patient is status post left frontotemporal craniotomy with craniectomy with
craniotomy with left anterior temporal lobe resection.  there is moderate
associated pneumocephalus and non organized fluid in the left middle cranial
fossa, which is rim by small quantities of patchy hemorrhagic products.  these
extend minimally into the adjacent left fronta
>
> \### response: false
>
> INFO:ludwig.models.llm:Decoded text inputs for the first example in batch: below is an instruction that describes a task, paired with an input that may provide further context. write a response that appropriately completes the request.
>
> \### instruction: based on the following medical notes, please predict whether the patient described is likely to have acute respiratory distress syndrome (ards). your prediction should be either 'true' if ards is likely, or 'false' if it is not likely.
>
> \### input: note 1: examination:  ct torso examination.
>
> indication:  ___ year old man with newly diagnosed left sided brain mass with
right homonymous hemianopsia  // evaluate for other lesions or masses
>
> technique:  single phase split bolus contrast: mdct axial images were acquired
through the chest, abdomen and pelvis following intravenous contrast
administration with split bolus technique.
oral contrast was administered.
coronal and sagittal reformations were performed and reviewed on pacs.
>
> dose:  total dlp (body) = 958 mgy-cm.
>
> comparison:  none.
>
> findings:
>
> there is a right apical mass, contiguous with the right upper mediastinum,
measuring approximately 8.8 x 5.4 x 5.9 cm (series 2, image 12, series 601b,
image 35).  the mass abuts the right subclavian vein without attenuation, and
contacts the svc without significant mass effect (series 2, image 17, 13). 
there is associated peripheral atelectasis (series 601b, image 36).  no
adjacent osseous invas
>
> \### response:
>
> INFO:ludwig.models.llm:Decoded generated output for the first example in batch: below is an instruction that describes a task, paired with an input that may provide further context. write a response that appropriately completes the request.
>
> \### instruction: based on the following medical notes, please predict whether the patient described is likely to have acute respiratory distress syndrome (ards). your prediction should be either 'true' if ards is likely, or 'false' if it is not likely.
>
> \### input: note 1: examination:  ct torso examination.
>
> indication:  ___ year old man with newly diagnosed left sided brain mass with
right homonymous hemianopsia  // evaluate for other lesions or masses
>
> technique:  single phase split bolus contrast: mdct axial images were acquired
through the chest, abdomen and pelvis following intravenous contrast
administration with split bolus technique.
oral contrast was administered.
coronal and sagittal reformations were performed and reviewed on pacs.
>
> dose:  total dlp (body) = 958 mgy-cm.
>
> comparison:  none.
>
> findings:
>
> there is a right apical mass, contiguous with the right upper mediastinum,
measuring approximately 8.8 x 5.4 x 5.9 cm (series 2, image 12, series 601b,
image 35).  the mass abuts the right subclavian vein without attenuation, and
contacts the svc without significant mass effect (series 2, image 17, 13). 
there is associated peripheral atelectasis (series 601b, image 36).  no
adjacent osseous invas
>
> \### response: false
>
> Prediction: 100%|██████████| 1/1 [00:03<00:00,  3.26s/it]
>
> INFO:ludwig.utils.tokenizers:Loaded HuggingFace implementation of meta-llama/Llama-2-7b-hf tokenizer
>
> WARNING:ludwig.utils.tokenizers:No padding token id found. Using eos_token as pad_token.
>
> Instruction: Based on the following medical notes, please predict whether the patient described is likely to have Acute Respiratory Distress Syndrome (ARDS). Your prediction should be either 'true' if ARDS is likely, or 'false' if it is not likely.
>
> Input: Note 1: EXAMINATION:   CT HEAD W/O CONTRAST Q111
>
> INDICATION:  ___ year old woman now s/p Left crani for tumor resection - please
perform PRIOR to ___  // ? interval changes / post operative hemorrhage
/stroke
>
> TECHNIQUE:  Multidetector CT images of the head were obtained without
intravenous contrast.  Sagittal and coronal reformations were also performed.
>
> DOSE:  Acquisition sequence:
>
>  1) Sequenced Acquisition 16.0 s, 16.5 cm; CTDIvol = 48.6 mGy (Head) DLP =
802.7 mGy-cm.
>
> Total DLP (Head) = 803 mGy-cm.
>
> COMPARISON:  Head CT is available from ___ and more recent MR from
the prior day.
>
> FINDINGS:
>
> Patient is status post left frontotemporal craniotomy with craniectomy with
craniotomy with left anterior temporal lobe resection.  There is moderate
associated pneumocephalus and non organized fluid in the left middle cranial
fossa, which is rim by small quantities of patchy hemorrhagic products.  These
extend minimally into the adjacent left fronta
Generated Output: false
>
> Instruction: Based on the following medical notes, please predict whether the patient described is likely to have Acute Respiratory Distress Syndrome (ARDS). Your prediction should be either 'true' if ARDS is likely, or 'false' if it is not likely.
Input: Note 1: EXAMINATION:  CT torso examination.
>
> INDICATION:  ___ year old man with newly diagnosed left sided brain mass with right homonymous hemianopsia  // Evaluate for other lesions or masses
>
> TECHNIQUE:  Single phase split bolus contrast: MDCT axial images were acquired
through the chest, abdomen and pelvis following intravenous contrast
administration with split bolus technique.
Oral contrast was administered.
Coronal and sagittal reformations were performed and reviewed on PACS.
>
> DOSE:  Total DLP (Body) = 958 mGy-cm.
>
> COMPARISON:  None.
>
> FINDINGS:
>
> There is a right apical mass, contiguous with the right upper mediastinum,
measuring approximately 8.8 x 5.4 x 5.9 cm (series 2, image 12, series 601b,
image 35).  The mass abuts the right subclavian vein without attenuation, and
contacts the SVC without significant mass effect (series 2, image 17, 13). 
There is associated peripheral atelectasis (series 601b, image 36).  No
adjacent osseous invas
Generated Output: false
>
> /usr/local/lib/python3.10/dist-packages/ludwig/features/feature_utils.py:102: RuntimeWarning: divide by zero encountered in log
  return np.sum(np.log(sequence_probabilities))



## Save Trained Model Artifacts To HuggingFace

Now that we have a fine-tuned model, we can export the model weights to HuggingFace hub so we can use them later. Ludwig supports uploading model weights directly to HuggingFace Hub via the upload Ludwig command.

```
!ludwig upload hf_hub --repo_id <hf_user_name>/<repo_name> --model_path <top_level_model_directory>
```


The model-path can be seen at the end of training/fine-tuning. You need to get a Huggingface Write API to upload

```
!ludwig upload hf_hub --repo_id MomochiKyaru/example --model_path /content/results/api_experiment_run
```

> _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
>
>  _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
>
>  _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
>
> _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
>
> _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
> 
> A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
> 
>     Setting a new token will erase the existing one.
> 
>     To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
> 
> Token:
> 
> Add token as git credential? (Y/n) n
> 
> Token is valid (permission: write).
> 
> Your token has been saved to /root/.cache/huggingface/token
> 
> Login successful
> 
> Traceback (most recent call last):
> 
>   File "/usr/local/bin/ludwig", line 8, in <module>

>     sys.exit(main())
> 
>   File "/usr/local/lib/python3.10/dist-packages/ludwig/cli.py", line 191, in main
> 
>     CLI()
> 
>   File "/usr/local/lib/python3.10/dist-packages/ludwig/cli.py", line 71, in __init__
> 
>     getattr(self, args.command)()
> 
>   File "/usr/local/lib/python3.10/dist-packages/ludwig/cli.py", line 186, in upload
> 
>     upload.cli(sys.argv[2:])
> 
>   File "/usr/local/lib/python3.10/dist-packages/ludwig/upload.py", line 125, in cli
> 
>     upload_cli(**vars(args))
> 
>   File "/usr/local/lib/python3.10/dist-packages/ludwig/upload.py", line 56, in upload_cli
> 
>     hub.upload(
> 
>   File "/usr/local/lib/python3.10/dist-packages/ludwig/utils/upload_utils.py", line 177, in upload
> 
>     HuggingFaceHub._validate_upload_parameters(
> 
>   File "/usr/local/lib/python3.10/dist-packages/ludwig/utils/upload_utils.py", line 107, in _validate_upload_parameters
> 
>     raise Exception(
> 
> Exception: Model artifacts not found at /content/results/api_experiment_run/model/model_weights. It is possible that model at '/content/results/api_experiment_run' hasn't been trained yet, or something went wrong during training where the model's weights were not saved.

If you want to store your weights locally, you may also download it through colab.

```
!zip -r /content/file.zip /content/results/api_experiment_run
```
> adding: content/results/api_experiment_run/ (stored 0%)
> 
> adding: content/results/api_experiment_run/description.json (deflated 80%)
> 
> adding: content/results/api_experiment_run/model/ (stored 0%)
> 
> adding: content/results/api_experiment_run/model/model_hyperparameters.json (deflated 88%)
> 
> adding: content/results/api_experiment_run/model/logs/ (stored 0%)
> 
> adding: content/results/api_experiment_run/model/logs/training/ (stored 0%)
> 
> adding: content/results/api_experiment_run/model/logs/training/events.out.tfevents.1696539509.61bd12dab543.206.0 (deflated 80%)
> 
> adding: content/results/api_experiment_run/model/logs/test/ (stored 0%)
> 
> adding: content/results/api_experiment_run/model/logs/test/events.out.tfevents.1696539509.61bd12dab543.206.2 (deflated 9%)
> 
> adding: content/results/api_experiment_run/model/logs/validation/ (stored 0%)
> 
> adding: content/results/api_experiment_run/model/logs/validation/events.out.tfevents.1696539509.61bd12dab543.206.1 (deflated 9%)
> 
> adding: content/results/api_experiment_run/model/training_set_metadata.json (deflated 83%)
> 
> adding: content/results/api_experiment_run/model/training_checkpoints/ (stored 0%)
> 
> adding: content/results/api_experiment_run/model/training_checkpoints/latest.ckpt (deflated 8%)
> 
> adding: content/results/api_experiment_run/model/training_progress.json (deflated 73%)
> 


```
from google.colab import files

files.download('REPLACE_MODEL_PATH_HERE')
```

## Load saved model from Huggingface and Inference

```
# Load the uploaded weights
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Default quantization config for base model in Ludwig AI
quantization_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype= torch.float16,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_use_double_quant= True,
    load_in_4bit=True,
    quant_method="bitsandbytes"
)

base_model = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype = torch.float16, quantization_config = quantization_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(base_model)
lora_config = PeftConfig.from_pretrained("HUGGINGFACE_ID/HUGGINGFACE_REPO")
model = PeftModel.from_pretrained(model, "HUGGINGFACE_ID/HUGGINGFACE_REPO", config = lora_config)

# Generate the instruction prompt for the fine-tuned model. Ludwig automatically transform inputs into lower cases, so we will also convert the prompt to lower case
def generate_prompt(instruction, input):
  prompt_template = f"""below is an instruction that describes a task, paired with an input that may provide further context. write a response that appropriately completes the request.
### instruction: {instruction}
### input: {input}
### response:"""
  return prompt_template.lower()

# use this function to get generated content
def generate_text(instruction, input):
  prompt = generate_prompt(instruction, input)
  inputs = tokenizer(prompt, return_tensors="pt")
  with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
    generated_text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].split("### response:")[-1].strip()
  return generated_text

generate_text('example instruction', 'example input')

```

## Task: Optimize the dummy LLM

Once you are familiar with fine-tuning LLM, you need to improve the model. Consider the directions below to further refine and leverage your model. Search for more informations on the internet and use LLM applications to increase your productivity (e.g. ChatGPT, github copilot)

### Leverage Domain-Specific Pre-Trained LLMs

- Explore Specialized LLMs: Investigate LLMs trained on domain-specific data like scholarly articles, journals. Consider using BioGPT, PubmedBERT, etc.Huggingface offer access to numerous domain-focused LLMs to cater to specific needs.
  
### Prompt Engineering:

- Instruction Design: Develop effective instruction prompts to guide the LLM in generating desired responses.
  
- Testing with ChatGPT: Utilize ChatGPT for testing your prompts, evaluating their efficacy in generating relevant outputs.
  
- Managing Prompt Length: Keep in mind that the length of the prompt impacts the maximum allowable input string length. Consider exploring various prompt engineering methods to optimize results.

### Input Note Truncation:
- Exploring Truncation Methods: Explore optimal methods to truncate input notes, ensuring preservation of key information.
  
- Utilizing Summarization: Employ LLMs to summarize notes, enabling the extraction of maximum information while adhering to input length constraints.
  
### Text Embedding:
- Text Embedding with LLMs: If computational resources are constrained, leverage LLMs to convert natural language notes into text embeddings (feature vectors) using relatively lower computational power.
  
- Project 1 Learnings Application: Apply the insights and techniques learned from Project 1 to train a classification model, using the derived embedding features to improve or refine model outcomes.
- 
## Dealing with imbalanced dataset:

- Optimizing of Output Type and Loss Function: With 8000 negative and approximately 100 positive samples, the dataset is notably imbalanced. Instead of a straightforward answer-generation task, guide the LLM to predict a boolean and adopt a loss function that heightens focus on positive samples.
  
- Data Augmentation: Contemplate enhancing the dataset by incorporating additional positive samples, mitigating imbalance and possibly improving model learning.
  
## Requirements

Save your predictions to the test set as a numpy array and dump to a pickle file 'prediction.pkl'. Submit your predict and report you efforts in optimizing the LLM. Using LLM API services like OpenAI API is not allowed(for fairness reason). You may train models on your own GPU, but peak GPU RAM usage should not exceed 15GB.

```
# make predictions on df test

df_test = pd.read_pickle("/content/drive/MyDrive/project2_test.pkl")

df_test = df.fillna("")
```
