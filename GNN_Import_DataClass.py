import pandas as pd 
import uproot4

class DataClass:

    def __init__(self, Path, Tree, MC_Channel, Variables, WeightList, Name, Cuts):
        self.Path            = Path
        self.Tree            = Tree
        self.MC_Channel      = MC_Channel
        self.Variables       = Variables
        self.WeightList      = WeightList
        self.Name            = Name
        self.Cuts            = Cuts


    def GetGNNInput(self):
        
        print(" "*19 + self.Name.ljust(15, " "), end="")              
        
        variables = self.Variables[:]
        variables.extend(self.WeightList)

        for i, path in enumerate(self.Path):
            if self.Cuts == '':
                selection = self.MC_Channel
            elif self.MC_Channel == '':
                selection = self.Cuts
            else:
                selection = "(" + self.MC_Channel + ") & (" + self.Cuts + ")"
        
        DataFrames = []
        for path in self.Path:
            root   = uproot4.open(path)
            tree   = root[self.Tree]
            Dict   = tree.arrays(cut=selection, expressions=variables, library="np")
            DataFrames.append(pd.DataFrame.from_dict(Dict))
                        
        DataFrame = pd.concat(DataFrames, ignore_index=True)
        DataFrame = DataFrame.rename(columns={self.WeightList[0]: 'weights'})

        print(str(DataFrame.shape[0]).rjust(6, " "))

        return DataFrame
            
      

def Init(FilePath, Variables, Cuts=True):


    # File Paths  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
    Path_tttt       = [FilePath + 'mc16a/2lss3lge1mv2c10j/4tops.root',        FilePath + 'mc16d/2lss3lge1mv2c10j/4tops.root',        FilePath + 'mc16e/2lss3lge1mv2c10j/4tops.root']
    Path_ttW        = [FilePath + 'mc16a/2lss3lge1mv2c10j/ttWSherpa.root',    FilePath + 'mc16d/2lss3lge1mv2c10j/ttWSherpa.root',    FilePath + 'mc16e/2lss3lge1mv2c10j/ttWSherpa.root']
    Path_ttWW       = [FilePath + 'mc16a/2lss3lge1mv2c10j/ttWW.root',         FilePath + 'mc16d/2lss3lge1mv2c10j/ttWW.root',         FilePath + 'mc16e/2lss3lge1mv2c10j/ttWW.root'] 
    Path_ttZ        = [FilePath + 'mc16a/2lss3lge1mv2c10j/ttZ.root',          FilePath + 'mc16d/2lss3lge1mv2c10j/ttZ.root',          FilePath + 'mc16e/2lss3lge1mv2c10j/ttZ.root',
                       FilePath + 'mc16a/2lss3lge1mv2c10j/ttll.root',         FilePath + 'mc16d/2lss3lge1mv2c10j/ttll.root',         FilePath + 'mc16e/2lss3lge1mv2c10j/ttll.root']
    Path_ttH        = [FilePath + 'mc16a/2lss3lge1mv2c10j/ttH.root',          FilePath + 'mc16d/2lss3lge1mv2c10j/ttH.root',          FilePath + 'mc16e/2lss3lge1mv2c10j/ttH.root']
    Path_vjets      = [FilePath + 'mc16a/2lss3lge1mv2c10j/vjets.root',        FilePath + 'mc16d/2lss3lge1mv2c10j/vjets.root',        FilePath + 'mc16e/2lss3lge1mv2c10j/vjets.root']
    Path_vv         = [FilePath + 'mc16a/2lss3lge1mv2c10j/vv.root',           FilePath + 'mc16d/2lss3lge1mv2c10j/vv.root',           FilePath + 'mc16e/2lss3lge1mv2c10j/vv.root']
    Path_singletop  = [FilePath + 'mc16a/2lss3lge1mv2c10j/single-top.root',   FilePath + 'mc16d/2lss3lge1mv2c10j/single-top.root',   FilePath + 'mc16e/2lss3lge1mv2c10j/single-top.root'] 
    Path_others     = [FilePath + 'mc16a/2lss3lge1mv2c10j/ttt.root',          FilePath + 'mc16d/2lss3lge1mv2c10j/ttt.root',          FilePath + 'mc16e/2lss3lge1mv2c10j/ttt.root',
                       FilePath + 'mc16a/2lss3lge1mv2c10j/vv.root',           FilePath + 'mc16d/2lss3lge1mv2c10j/vv.root',           FilePath + 'mc16e/2lss3lge1mv2c10j/vv.root',
                       FilePath + 'mc16a/2lss3lge1mv2c10j/vvv.root',          FilePath + 'mc16d/2lss3lge1mv2c10j/vvv.root',          FilePath + 'mc16e/2lss3lge1mv2c10j/vvv.root',
                       FilePath + 'mc16a/2lss3lge1mv2c10j/vh.root',           FilePath + 'mc16d/2lss3lge1mv2c10j/vh.root',           FilePath + 'mc16e/2lss3lge1mv2c10j/vh.root']
    Path_ttbar_else = [FilePath + 'mc16a/2lss3lge1mv2c10j/ttbar.root',        FilePath + 'mc16d/2lss3lge1mv2c10j/ttbar.root',        FilePath + 'mc16e/2lss3lge1mv2c10j/ttbar.root',
                       FilePath + 'mc16a/2lss3lge1mv2c10j/single-top.root',   FilePath + 'mc16d/2lss3lge1mv2c10j/single-top.root',   FilePath + 'mc16e/2lss3lge1mv2c10j/single-top.root']


    # ROOT Channels ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
    tree                    = 'nominal_Loose'
    MC_Channel_tttt         = "(mcChannelNumber==412115)"
    MC_Channel_NLO          = "(mcChannelNumber==412043)"
    MC_Channel_ttW          = ""
    MC_Channel_ttWW         = "(mcChannelNumber==410081)"
    MC_Channel_ttZ          = "((mcChannelNumber==410156)|(mcChannelNumber==410157)|(mcChannelNumber==410218)|(mcChannelNumber==410219)|(mcChannelNumber==410220)|(mcChannelNumber==410276)|(mcChannelNumber==410277)|(mcChannelNumber==410278))"
    MC_Channel_ttH          = "((mcChannelNumber==346345)|(mcChannelNumber==346344))"
    MC_Channel_vjets        = ""
    MC_Channel_vv           = ""
    MC_Channel_singletop    = "(event_BkgCategory==0) & ((mcChannelNumber==410646)|(mcChannelNumber==410647)|(mcChannelNumber==410560)|(mcChannelNumber==410408))"
    MC_Channel_others       = "((mcChannelNumber>=364242)&(mcChannelNumber<=364249)) | (mcChannelNumber==342284) | (mcChannelNumber==342285) | (mcChannelNumber==304014)"
    MC_Channel_ttbar_Qmis   = "(((mcChannelNumber==410658)|(mcChannelNumber==410659)|(mcChannelNumber==410646)|(mcChannelNumber==410647)|(mcChannelNumber==410644)|(mcChannelNumber==410645)) | ((mcChannelNumber>=407342)&(mcChannelNumber<=407344)) | ((mcChannelNumber==410470)&(GenFiltHT/1000.<600))) & (event_BkgCategory==1)"
    MC_Channel_ttbar_CO     = "(((mcChannelNumber==410658)|(mcChannelNumber==410659)|(mcChannelNumber==410646)|(mcChannelNumber==410647)|(mcChannelNumber==410644)|(mcChannelNumber==410645)) | ((mcChannelNumber>=407342)&(mcChannelNumber<=407344)) | ((mcChannelNumber==410470)&(GenFiltHT/1000.<600))) & ((event_BkgCategory==2)|(event_BkgCategory==3))"
    MC_Channel_ttbar_HF     = "(((mcChannelNumber==410658)|(mcChannelNumber==410659)|(mcChannelNumber==410646)|(mcChannelNumber==410647)|(mcChannelNumber==410644)|(mcChannelNumber==410645)) | ((mcChannelNumber>=407342)&(mcChannelNumber<=407344)) | ((mcChannelNumber==410470)&(GenFiltHT/1000.<600))) & ((event_BkgCategory==4)|((((mcChannelNumber>=407342)&(mcChannelNumber<=407344))|((mcChannelNumber==410470)&(GenFiltHT/1000.<600)))&(event_BkgCategory==0)))"
    MC_Channel_ttbar_light  = "(((mcChannelNumber==410658)|(mcChannelNumber==410659)|(mcChannelNumber==410646)|(mcChannelNumber==410647)|(mcChannelNumber==410644)|(mcChannelNumber==410645)) | ((mcChannelNumber>=407342)&(mcChannelNumber<=407344)) | ((mcChannelNumber==410470)&(GenFiltHT/1000.<600))) & (event_BkgCategory==5)"
    MC_Channel_ttbar_others = "(((mcChannelNumber==410658)|(mcChannelNumber==410659)|(mcChannelNumber==410646)|(mcChannelNumber==410647)|(mcChannelNumber==410644)|(mcChannelNumber==410645)) | ((mcChannelNumber>=407342)&(mcChannelNumber<=407344)) | ((mcChannelNumber==410470)&(GenFiltHT/1000.<600))) & (event_BkgCategory==6)"
 
                
    # Weights ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
    WeightLO  = ["sig_weight"]
    WeightNLO = ["((36207.7*(runNumber==284500)+44307.4*(runNumber==300000)+(runNumber==310000)*58450.1)*weight_normalise*weight_pileup*weight_jvt*weight_mc*weight_leptonSF*weight_bTagSF_MV2c10_Continuous_CDI20190730)"]

    # WeightLO  = ["((weight_normalise*weight_mcweight_normalise[85]/weight_mcweight_normalise[0]*weight_pileup*weight_jvt*mc_generator_weights[85]*weight_leptonSF*weight_bTagSF_MV2c10_Continuous_CDI20190730)*(36207.7*(runNumber==284500)+44307.4*(runNumber==300000)+(runNumber==310000)*58450.1))"]
    # Problematic Variables: weight_mcweight_normalise[85], weight_mcweight_normalise[0], mc_generator_weights[85]

    # Cuts  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
    if Cuts == False:
        Cuts = ''
    else:
        Cuts = "(nBTags_MV2c10_77>=2) & (nJets>=6) & (HT_all>500000) & ((SSee_passECIDS==1) | (SSem_passECIDS==1) | (SSmm==1) | (eee_Zveto==1) | (eem_Zveto==1) | (emm_Zveto==1) | (mmm_Zveto==1))"



    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
    #   S a v e   D a t a   a s   a   D a t a C l a s s   I n s t a n c e                                                                               #
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

    # Class Instance     | DataClass( Path,              Tree,   MC_Channel,                  Variables,     WeightList,    Name,             Cuts)
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    Sample_tttt_LO       = DataClass( Path_tttt,         tree,   MC_Channel_tttt,             Variables,     WeightLO,      'tttt_LO',        Cuts      )
    Sample_tttt_NLO      = DataClass( Path_tttt,         tree,   MC_Channel_NLO,              Variables,     WeightNLO,     'tttt_NLO',       Cuts      )
    Sample_ttW           = DataClass( Path_ttW,          tree,   MC_Channel_ttW,              Variables,     WeightNLO,     'ttW',            Cuts      )
    Sample_ttWW          = DataClass( Path_ttWW,         tree,   MC_Channel_ttWW,             Variables,     WeightNLO,     'ttWW',           Cuts      )
    Sample_ttZ           = DataClass( Path_ttZ,          tree,   MC_Channel_ttZ,              Variables,     WeightNLO,     'ttZ',            Cuts      )
    Sample_ttH           = DataClass( Path_ttH,          tree,   MC_Channel_ttH,              Variables,     WeightNLO,     'ttH',            Cuts      )
    Sample_vjets         = DataClass( Path_vjets,        tree,   MC_Channel_vjets,            Variables,     WeightNLO,     'vjets',          Cuts      )
    Sample_vv            = DataClass( Path_vv,           tree,   MC_Channel_vv,               Variables,     WeightNLO,     'vv',             Cuts      )
    Sample_singletop     = DataClass( Path_singletop,    tree,   MC_Channel_singletop,        Variables,     WeightNLO,     'singletop',      Cuts      )
    Sample_others        = DataClass( Path_others,       tree,   MC_Channel_others,           Variables,     WeightNLO,     'others',         Cuts      )
    Sample_ttbar_Qmis    = DataClass( Path_ttbar_else,   tree,   MC_Channel_ttbar_Qmis,       Variables,     WeightNLO,     'ttbar_Qmis',     Cuts      )
    Sample_ttbar_CO      = DataClass( Path_ttbar_else,   tree,   MC_Channel_ttbar_CO,         Variables,     WeightNLO,     'ttbar_CO',       Cuts      )
    Sample_ttbar_HF      = DataClass( Path_ttbar_else,   tree,   MC_Channel_ttbar_HF,         Variables,     WeightNLO,     'ttbar_HF',       Cuts      )
    Sample_ttbar_light   = DataClass( Path_ttbar_else,   tree,   MC_Channel_ttbar_light,      Variables,     WeightNLO,     'ttbar_light',    Cuts      )
    Sample_ttbar_others  = DataClass( Path_ttbar_else,   tree,   MC_Channel_ttbar_others,     Variables,     WeightNLO,     'ttbar_others',   Cuts      )



    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
    #   C o m b i n e   C l a s s   I n s t a n c e s   i n   a   L i s t   o f   S a m p l e s                                                         #
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

    
    # List_Samples = [Sample_tttt_LO,          Sample_tttt_NLO,           Sample_ttW,           Sample_ttWW,          Sample_ttZ,
    #                 Sample_ttH,              Sample_vjets,              Sample_vv,            Sample_singletop,     Sample_others,
    #                 Sample_ttbar_Qmis,       Sample_ttbar_CO,           Sample_ttbar_HF,      Sample_ttbar_light,   Sample_ttbar_others]  

    List_Samples = [Sample_vjets,              Sample_vv,     Sample_ttbar_Qmis,]

    
    return List_Samples
