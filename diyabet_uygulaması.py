# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:45:08 2021

@author: SABRİ GENÇ
"""


# %%

from pandas import read_csv
from numpy import unique
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from numpy import ravel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from mlxtend.evaluate import bias_variance_decomp
import os 

class diyabet():
    
    def __init__(self):
        
        self.veri_okuma()
        self.tek_kolon_veri_taraması()
        self.tekrarlanan_veri_taraması()
        self.aykırı_veri_taraması()
        self.özellik_seçimi()
        self.ileri_seviye_dönüşümler()
        self.algoritma_seçimi()
        self.algoritma_hiperparametreleri()
        self.cross_validation()
        self.asırı_ögrenim_analizi()
        self.sonuc()
        
    def veri_okuma(self):
        
        self.veri=read_csv("diabetes.csv")
        self.x,self.y=self.veri.iloc[:,:-1],self.veri.iloc[:,-1]
        
    
    def tek_kolon_veri_taraması(self):
        
        values=self.x.nunique()
        done=bool(0)
    
        done=bool(0)
        if not done:
            for i,j in enumerate(values):
                if j==1:
                    self.x.drop(axis=1,inplace=True,columns=i)
                else:
                    pass
        
        self.x=self.x.values
    
    def tekrarlanan_veri_taraması(self):
        
        self.x=DataFrame(self.x)
        self.y=DataFrame(self.y)
        self.veri=concat([self.x,self.y],axis=1)
        
        done=bool(0)
        if not done:
            self.veri.drop_duplicates(inplace=True)
            self.x,self.y=self.veri.iloc[:,:-1],self.veri.iloc[:,-1]
    
         
    def aykırı_veri_taraması(self):
        
        model=KNeighborsClassifier()
        x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.33,random_state=0)
        
        done=bool(0)
        if not done:
            lof=LocalOutlierFactor()
            aykırı_deger=lof.fit_predict(self.veri)
            for i,j in enumerate(aykırı_deger):
                if j!=1:
                    self.veri.drop(axis=0,inplace=True,index=i)
                
                else:
                    pass
            self.x,self.y=self.veri.iloc[:,:-1],self.veri.iloc[:,-1]
            x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.33,random_state=0)
            model.fit(x_train,y_train)
            tahmin=model.predict(x_test)
            
            cm=confusion_matrix(y_test,tahmin)
            #print("CM1:",cm)
        
        else:
            self.x,self.y=self.veri.iloc[:,:-1],self.veri.iloc[:,-1]
            x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.33,random_state=0)
            model.fit(x_train,y_train)
            tahmin=model.predict(x_test)
            
            cm=confusion_matrix(y_test,tahmin)
            #print("CM1:",cm)
         
        
    def özellik_seçimi(self):
        
              
        done=bool(-1)
        if not done:
            x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.33,random_state=0)
            fs=SelectKBest(score_func=f_classif,k="all")
            fs.fit(x_train,y_train)
            for i in range(len(fs.scores_)):
                print("%d.Feature %.3f"%(i,fs.scores_[i]))
                
            plt.bar([i for i in range(len(fs.scores_))],fs.scores_)
            plt.tight_layout()
            plt.show()
        
            x_train=fs.transform(x_train)
            x_test=fs.transform(x_test)
            
            model=KNeighborsClassifier()
            model.fit(x_train,y_train)
            tahmin=model.predict(x_test)
        
            cm=confusion_matrix(y_test,tahmin)
            print("cm: {}".format(cm))
        else:
            fs=SelectKBest(score_func=mutual_info_classif,k=7)
            fs.fit(self.x,self.y)
            self.x=fs.transform(self.x)
            
        
        self.x=DataFrame(self.x,index=range(self.x.shape[0]))
        self.y=DataFrame(self.y.values,index=range(self.x.shape[0]),columns=[i for i in range(5,6)])
        self.veri=concat([self.x,self.y],axis=1)
        
        return self.veri
        
        
    def ileri_seviye_dönüşümler(self):
        
        dönüşümler=["POWER TRANSFORMER","QUANTİLE TRANSFORMER"]
        done=bool(-1)
        if not done:
            model=KNeighborsClassifier()
            scl1=StandardScaler()
            pipeline=Pipeline(steps=[("s",scl1),("m",model)])
            cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
            skor=cross_val_score(pipeline,self.x,self.y,scoring="accuracy",cv=cv,n_jobs=-1)
            
            for i in range(1,3):
                if i==1:
                    method=["yeo-johnson"]
                    for j in method:
                        scl1=PowerTransformer(method=j)
                        pipeline=Pipeline(steps=[("s",scl1),("m",model)])
                        cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
                        skor=cross_val_score(pipeline,self.x,self.y,scoring="accuracy",cv=cv,n_jobs=-1)
                        print("method: %s scl1--> mean:%.3f --- std: %.3f"%(j,mean(skor),std(skor)))
                
                elif i==2:
                    method=["uniform","normal"]
                    n_quan=[i for i in range(5,25,5)]
 
                    for i in method:
                        print("--"*10)
                        for j in n_quan:
                            scl2=QuantileTransformer(n_quantiles=j,output_distribution=i,random_state=0)
                            pipeline=Pipeline(steps=[("s",scl2),("m",model)])
                            cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
                            skor=cross_val_score(pipeline,self.x,self.y,scoring="accuracy",cv=cv,n_jobs=-1)
                            print("n_quan = %d -- method: %s scl2--> mean:%.3f --- std: %.3f"%(j,i,mean(skor),std(skor)))
                    
        else:
            trans=QuantileTransformer(n_quantiles=5,output_distribution="normal")
            self.x,self.y=self.veri.iloc[:,:-1],self.veri.iloc[:,-1]
            self.x=trans.fit_transform(self.x)
        
        self.x=DataFrame(self.x)
        self.y=DataFrame(self.y,index=range(self.x.shape[0]),columns=[i for i in range(5,6)])
        
        self.veri=concat([self.x,self.y],axis=1).values
        
                
    def algoritma_seçimi(self):
        
        self.y=ravel(self.y)
        self.x=self.x.values
        
        x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.33,random_state=0)
        
        models=[KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),
                GradientBoostingClassifier(),GaussianNB(),LogisticRegression(),SVC()]
        done=bool(-1)
        if not done:
            for i in models:
                i.fit(x_train,y_train)
                tahmin=i.predict(x_test)
                accur=accuracy_score(y_test,tahmin)
                print("----"*5)
                print("%s models accuracy= %.3f"%(i,accur))
    
    def algoritma_hiperparametreleri(self):
        
        
        s=[{"n_estimators":[1,2,3,4,5,6,7,8],"criterion":["gini","entropy"],
           "max_depth":[1,2,3,4,5,6,7,8],"max_features":["auto","sqrt","log2"]}]
        
        done=bool(-1)
        if not done:
            cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
            model=RandomForestClassifier()
            gs=GridSearchCV(model,param_grid=s,scoring="accuracy",n_jobs=-1,cv=cv)
            gs.fit(self.x,self.y)
            
            print("best_params: %s"%(gs.best_params_))
            print("best_score: %.3f"%(gs.best_score_))
            
        
    def cross_validation(self):
        
        model=RandomForestClassifier(n_estimators=7,criterion="entropy",max_depth=5,max_features="sqrt")
        kfold=KFold(n_splits=10,shuffle=True,random_state=0)
        score=cross_val_score(model,self.x,self.y,scoring="accuracy",cv=kfold,n_jobs=-1)
        
        #print("mean: %.3f -- std: %.3f"%(mean(score),std(score)))
      
     
   
    def asırı_ögrenim_analizi(self):
    
     
        x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.33,random_state=0)  
        acc_train=list()
        acc_test=list()
        done=bool(-1)
        values=[i for i in range(1,51)]
        if not done:
            for i in values:
                print("--------"*8)
                model=RandomForestClassifier(n_estimators=i,criterion="entropy",max_depth=i,max_features="sqrt")
                model.fit(x_train,y_train)
                
                tahmin_xtrain=model.predict(x_train)
                train_acc=accuracy_score(y_train,tahmin_xtrain)
                acc_train.append(train_acc)
                
                test_xtest=model.predict(x_test)
                test_acc=accuracy_score(y_test,test_xtest)
                acc_test.append(test_acc)
                
                mse,bias,variance=bias_variance_decomp(model,x_train, y_train, x_test, y_test, loss="mse",
                                                       num_rounds=200,random_seed=1)
                
                print("%d.train= %.3f -- test= %.3f"%(i,train_acc,test_acc))
                print("%d MSE=%.3f --- Bias= %.3f --- Variance= %.3f"%(i,mse,bias,variance))
            
            plt.plot(values,acc_train,"-o",label="Train")
            plt.plot(values,acc_test,"-o",label="Test")
            plt.title("AŞIRI ÖĞRENİM ANALİZİ FERMUAR METODU")
            plt.tight_layout()
            plt.legend()
            plt.show()
            
        
   
    def sonuc(self):
        print("\n")
        print("""!!!!UYARI!!!! SİSTEM DİYABET HASTA VERİSİ İLE EĞİTİMİ YAPILDI VE FARKLI BİR DİYABET VERİSİ İLE DE TESTİ GERÇEKLEŞTİRİLDİ SİSTEM SİZE KAÇ KİŞİNİN TAHLİL SONUCUNU SİSTEME GÖSTERMEK İSTEDİĞİNİZİ SORMAKTADIR. HARİCİ BAŞKA VERİLERLE TEST EDİLMEK İSTENİRSE VE O VERİDE EKSİK VERİLER OLDUĞU TAKTİRDE SİSTEM  HATA VERİCEKTİR FAKAT EKSİK VERİLERİ ÇÖZMEK İÇİN  ALGORİTMAYA EKLEME YAPILARAK EKSİK VERİLERE KARŞI ÇÖZÜM ÜRETMESİ SAĞLANABİLİR BU SİSTEMDE KULLANILAN VERİLERDE EKSİK VERİ BULUNMAMAKTADIR.
              
              SİSTEME VERİ GİRİŞİ YAPILACAKSA VERİ DOSYASI KODUN KAYNAK DİZİNİNDE BULUNMASI GEREKMEKTEDİR AYRICA YÜKLEMEK İSTEDİĞİNİZ VERİ 'diabets.csv ve test_diyabet.csv' VERİLERİNE BENZER VE O DÜZENDE(FORMATTA) OLMASINA DİKKAT EDİNİZ AKSİ TAKTİRDE SİSTEM HATA VERECEKTİR 
                      
              ----->veri girmek için 1 yazıp enter tuşuna basınız...
              ----->veri girmeden devam etmek için ise 0 yazıp enter tuşuna basınız...
              
              """)
              
        veri_seçme=int(input("Yapmak istediginiz işlemi belirtiniz="))
        
        if veri_seçme==1:
            dosya_adı=str(input("""Okunacak verinin adını tırnak içinde olmadan örneğin veri.csv şeklinde yazınız
                                (Büyük küçük harflere dikkat ediniz) ="""))    
            self.yeni_veri=read_csv(dosya_adı)
            self.x2,self.y2=self.yeni_veri.iloc[:,:-1],self.yeni_veri.iloc[:,-1]
        
        else:
            
            self.yeni_veri=read_csv("test_diyabet.csv")
            self.x2,self.y2=self.yeni_veri.iloc[:,:-1],self.yeni_veri.iloc[:,-1]
        
        self.x2=self.x2.values
        self.y2=self.y2.values
        
        done=bool(0)
        if not done:
            scl1=StandardScaler()
            self.x2=scl1.fit_transform(self.x2)
            
            trans=QuantileTransformer(n_quantiles=5,output_distribution="normal")
            self.x2=trans.fit_transform(self.x2)
            
            if done == 0:
                fs=SelectKBest(score_func=f_classif,k=7)
                fs.fit(self.x2,self.y2)
                self.x2=fs.transform(self.x2)
            
        
        x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.33,random_state=0)
        model=RandomForestClassifier(n_estimators=5,criterion="gini",max_depth=5,max_features="auto")
        model.fit(x_train,y_train)
        
        tahmin=model.predict(x_test)
        accur=accuracy_score(y_test,tahmin)
        print("----------\n")
        print("----------")
        print("----------")
        print("eski veriden aldıgı başarı--> mean: %.3f -- std: %.3f"%(mean(accur),std(accur)))
        
        yeni_tahmin=model.predict(self.x2)
        accur2=accuracy_score(self.y2,yeni_tahmin)
        print("yeni görmediği veriden aldıgı sonuc--> mean: %.3f -- std: %.3f"%(mean(accur2),std(accur2)))
        print("----------")
        print("----------")
        print("----------")
        print("-------\n")
        self.x2=DataFrame(self.x2)
        self.y2=DataFrame(self.y2,columns=[i for i in range(self.x2.shape[1],self.x2.shape[1]+1)])
        self.yeni_veri=concat([self.x2,self.y2],axis=1)
        
        self.dosya_dizin=os.getcwd()+"\hasta_sonuc.csv"
        self.cevap_dosya=os.getcwd()+"\cevap.csv"
        
        done= bool(0)
        rapor=list()
        print("\n"*2)
        if not done:
            cevap=int(input("KAÇ HASTA TAHLİL SONUCUNU SİSTEME GÖSTERMEK İSTERSİNİZ:"))
            if cevap>0:
                for i in range(cevap):
                    
                    sonuc=model.predict(self.x2.iloc[i:i+1,:])
                    if sonuc==1:
                        rapor.append("%d.kisi diyabet hastasidir"%(i+1))
                    
                    else:
                        rapor.append("%d.kisi diyabet hastasi degildir"%(i+1))
           
                  
                veri_hali=DataFrame(rapor,index=range(len(rapor)),columns=["HASTA SONUCLARI"])
                veri_hali.to_csv(self.dosya_dizin,index=False)
                print("\n")
                print("HASTA RAPOR SONUÇLARI KAYNAK KODU DİZİNİNDE hasta_sonuc.csv ADINDA EXCEL DOSYASI OLUŞTURULDU !!!")
                print("------------"*8)
                print("SİSTEMİN VERMESİ GEREK DOĞRU CEVAPLAR LİSTESİDE KODUN KAYNAK DİZİNİNDE 'cevap.csv' ADINDA OLUŞTURULDU !!!")
            else:
                print("\n")
                print("!!!!! HATA !!! HASTA SAYISI GİRİLMEDİ TEKRAR DENEYİNİZ. !!!!")
        
        self.y2.to_csv(self.cevap_dosya,index=False,header=None)
       
    
diyabet()




