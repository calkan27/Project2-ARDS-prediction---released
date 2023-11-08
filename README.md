# Project 2

Acute respiratory distress syndrome (ARDS) is a serious lung condition that causes low blood oxygen. In this project, you need to make a preidction about if the patient has ARDS based on their medical notes. Each patient may have multiple notes.

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

'''

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
  
  '''
  
Enable text wrapping so we don't have to scroll horizontally and create a function to flush CUDA cache.

  '''
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
    
  '''
