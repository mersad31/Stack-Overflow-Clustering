from __future__ import annotations

import time
from typing import Any
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, adjusted_rand_score, mean_absolute_error, silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from .bonus import run_bonus_analysis
from .config import project_path
from .features import split_multilabel
from .io_utils import sha256_file, utc_now_iso, write_json

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges=np.unique(np.quantile(reference,np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0],edges[-1]=-np.inf,np.inf
    a=np.histogram(reference,bins=edges)[0]/len(reference); b=np.histogram(current,bins=edges)[0]/len(current)
    a=np.clip(a,1e-6,None); b=np.clip(b,1e-6,None)
    return float(np.sum((b-a)*np.log(b/a)))


def mean_absolute_shap_values(values: Any, feature_count: int) -> np.ndarray:
    """Reduce SHAP outputs from supported binary/multiclass layouts to features."""
    if isinstance(values, list):
        result = np.mean(np.abs(np.stack(values, axis=0)), axis=(0, 1))
    else:
        array = np.asarray(values)
        if array.ndim == 3 and array.shape[1] == feature_count:
            result = np.mean(np.abs(array), axis=(0, 2))
        elif array.ndim == 3 and array.shape[2] == feature_count:
            result = np.mean(np.abs(array), axis=(0, 1))
        elif array.ndim == 2 and array.shape[1] == feature_count:
            result = np.mean(np.abs(array), axis=0)
        else:
            raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    if result.shape != (feature_count,):
        raise ValueError(f"SHAP reduction returned {result.shape}, expected {(feature_count,)}")
    return result

def run_phase3_analysis(config: dict[str,Any]) -> dict[str,Any]:
    started=time.perf_counter(); seed=int(config['project']['random_state'])
    processed=project_path(config,'processed_dir'); interim=project_path(config,'interim_dir')
    artifacts=project_path(config,'artifacts_dir'); tables=project_path(config,'tables_dir'); models=artifacts/'models'
    x=np.load(processed/'X_pca.npy',mmap_mode='r'); candidates=np.load(processed/'phase2_family_candidate_labels.npz')
    idx=candidates['sample_indices']; names=[n for n in candidates.files if n!='sample_indices']; labelings=[candidates[n] for n in names]
    n=len(idx); co=np.zeros((n,n),dtype=np.float32)
    for labels in labelings: co += labels[:,None]==labels[None,:]
    co/=len(labelings); distance=1-co
    rows=[]; consensus_labels={}
    sample=np.asarray(x[idx],dtype=np.float32)
    for k in range(2,7):
        labels=AgglomerativeClustering(n_clusters=k,metric='precomputed',linkage='average').fit_predict(distance)
        counts=np.bincount(labels); sil=float(silhouette_score(sample,labels))
        rows.append({'k':k,'silhouette':sil,'minimum_cluster_fraction':float(counts.min()/n),'clusters':k})
        consensus_labels[k]=labels
    selection=pd.DataFrame(rows); valid=selection[selection.minimum_cluster_fraction>=0.02]
    chosen_k=int(valid.loc[valid.silhouette.idxmax(),'k']); y_sample=consensus_labels[chosen_k]
    selection.to_csv(tables/'phase3_consensus_k_selection.csv',index=False,encoding='utf-8-sig')
    np.save(processed/'phase3_consensus_coassociation.npy',co,allow_pickle=False)

    train_i,test_i=train_test_split(np.arange(n),test_size=.25,stratify=y_sample,random_state=seed)
    assigner=RandomForestClassifier(n_estimators=180,min_samples_leaf=3,n_jobs=-1,random_state=seed,class_weight='balanced')
    assigner.fit(sample[train_i],y_sample[train_i]); validation_accuracy=float(accuracy_score(y_sample[test_i],assigner.predict(sample[test_i])))
    assigner.fit(sample,y_sample); labels=assigner.predict(x).astype(np.int16)
    np.save(processed/'phase3_consensus_labels.npy',labels,allow_pickle=False)
    centroids=np.vstack([np.asarray(x[labels==cluster]).mean(axis=0) for cluster in np.unique(labels)]).astype(np.float32)
    np.save(processed/'phase3_consensus_centroids.npy',centroids,allow_pickle=False)
    joblib.dump(assigner,models/'phase3_consensus_assigner.joblib')

    metadata=pd.read_parquet(processed/'respondent_metadata.parquet'); clean=pd.read_parquet(interim/'cleaned_cohort.parquet')
    metadata=metadata.copy(); metadata['cluster']=labels
    numeric=['ExperienceConsensus','TechnologyBreadth','YearsCodeNumeric','YearsCodeProNumeric','WorkExpNumeric','ConvertedCompYearly']
    profile_rows=[]
    for cluster,group in metadata.groupby('cluster'):
        row={'cluster':int(cluster),'respondents':len(group),'share_pct':100*len(group)/len(metadata)}
        for col in numeric: row[f'{col}_median']=float(group[col].median()) if group[col].notna().any() else np.nan
        for col in ['DevType','EdLevel','Employment','RemoteWork','AISelect','Country']:
            row[f'{col}_mode']=group[col].fillna('Missing').mode().iloc[0]
        profile_rows.append(row)
    profiles=pd.DataFrame(profile_rows)
    breadth_order=profiles.sort_values('TechnologyBreadth_median').cluster.tolist()
    profile_names={int(breadth_order[0]):'Focused-stack practitioners',int(breadth_order[-1]):'Broad-stack polyglots'}
    profiles.insert(1,'profile_name',profiles.cluster.map(profile_names))
    profiles.to_csv(tables/'phase3_cluster_profiles.csv',index=False,encoding='utf-8-sig')
    tech_rows=[]
    for cluster in sorted(np.unique(labels)):
        mask=labels==cluster
        for col in config['data']['technology_columns']:
            counts=clean.loc[mask,col].map(split_multilabel).explode().dropna().value_counts().head(8)
            for rank,(tech,count) in enumerate(counts.items(),1): tech_rows.append({'cluster':int(cluster),'domain':col.replace('HaveWorkedWith',''),'rank':rank,'technology':tech,'respondents':int(count),'pct_cluster':100*count/mask.sum()})
    pd.DataFrame(tech_rows).to_csv(tables/'phase3_cluster_top_technologies.csv',index=False,encoding='utf-8-sig')

    sil=silhouette_samples(sample,y_sample); cases=[]
    for cluster in sorted(np.unique(y_sample)):
        pos=np.flatnonzero(y_sample==cluster)
        within=cdist(sample[pos],sample[pos]); medoid=pos[np.argmin(within.sum(axis=1))]
        for kind,local in [('medoid',medoid),('exemplar',pos[np.argmax(sil[pos])]),('boundary',pos[np.argmin(sil[pos])])]:
            global_i=int(idx[local]); cases.append({'cluster':int(cluster),'kind':kind,'row_index':global_i,'ResponseId':int(metadata.iloc[global_i].ResponseId),'silhouette':float(sil[local])})
    pd.DataFrame(cases).to_csv(tables/'phase3_exemplars_boundaries.csv',index=False,encoding='utf-8-sig')

    tree_train,tree_test=train_test_split(np.arange(len(x)),test_size=.25,stratify=labels,random_state=seed)
    tree=DecisionTreeClassifier(max_depth=4,min_samples_leaf=100,class_weight='balanced',random_state=seed).fit(x[tree_train],labels[tree_train])
    tree_validation_accuracy=float(accuracy_score(labels[tree_test],tree.predict(x[tree_test])))
    tree.fit(x,labels)
    tree_training_accuracy=float(accuracy_score(labels,tree.predict(x))); joblib.dump(tree,models/'phase3_explanation_tree.joblib')
    feature_names=[f'PC{i}' for i in range(1,x.shape[1]+1)]
    (artifacts/'phase3_decision_rules.txt').write_text(export_text(tree,feature_names=feature_names),encoding='utf-8')
    importance=pd.DataFrame({'feature':feature_names,'importance':tree.feature_importances_}).sort_values('importance',ascending=False)
    try:
        import shap
        sv=shap.TreeExplainer(tree).shap_values(np.asarray(x[:1000]))
        values=mean_absolute_shap_values(sv,len(feature_names))
        shap_by_feature=dict(zip(feature_names,values))
        importance['shap_mean_abs']=importance['feature'].map(shap_by_feature)
        shap_status='completed'
    except Exception as exc:
        importance['shap_mean_abs']=importance['importance']; shap_status=f'fallback: {type(exc).__name__}'
    importance.to_csv(tables/'phase3_feature_importance_shap.csv',index=False,encoding='utf-8-sig')

    anomaly=np.empty(len(x),dtype=np.float32)
    for cluster in np.unique(labels):
        pos=np.flatnonzero(labels==cluster); model=IsolationForest(n_estimators=120,contamination=.01,n_jobs=-1,random_state=seed+int(cluster)).fit(x[pos]); anomaly[pos]=-model.score_samples(x[pos])
    metadata[['ResponseId','cluster']].assign(anomaly_score=anomaly).nlargest(max(1,len(x)//100),'anomaly_score').to_csv(tables/'phase3_cluster_anomalies.csv',index=False,encoding='utf-8-sig')

    comp=metadata.ConvertedCompYearly.notna() & (metadata.ConvertedCompYearly>0); pos=np.flatnonzero(comp); tr,te=train_test_split(pos,test_size=.25,random_state=seed)
    global_reg=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,n_jobs=-1,random_state=seed).fit(x[tr],np.log1p(metadata.ConvertedCompYearly.iloc[tr]))
    down=[{'model':'global','cluster':'all','test_rows':len(te),'mae_log':mean_absolute_error(np.log1p(metadata.ConvertedCompYearly.iloc[te]),global_reg.predict(x[te]))}]
    for cluster in np.unique(labels):
        ctr=tr[labels[tr]==cluster]; cte=te[labels[te]==cluster]
        if len(ctr)<100 or len(cte)<30: continue
        reg=RandomForestRegressor(n_estimators=80,min_samples_leaf=5,n_jobs=-1,random_state=seed+int(cluster)).fit(x[ctr],np.log1p(metadata.ConvertedCompYearly.iloc[ctr]))
        down.append({'model':'cluster_conditional','cluster':int(cluster),'test_rows':len(cte),'mae_log':mean_absolute_error(np.log1p(metadata.ConvertedCompYearly.iloc[cte]),reg.predict(x[cte])),'global_mae_same_rows':mean_absolute_error(np.log1p(metadata.ConvertedCompYearly.iloc[cte]),global_reg.predict(x[cte]))})
    pd.DataFrame(down).to_csv(tables/'phase3_downstream_compensation.csv',index=False,encoding='utf-8-sig')

    composition=[]
    for col in ['EdLevel','RemoteWork','Country','AISelect']:
        tab=pd.crosstab(metadata.cluster,metadata[col].fillna('Missing'),normalize='index')
        for cluster,row in tab.iterrows():
            for category,value in row.items(): composition.append({'attribute':col,'cluster':cluster,'category':category,'share':value})
    pd.DataFrame(composition).to_csv(tables/'phase3_composition_audit.csv',index=False,encoding='utf-8-sig')

    sensitivity=[]
    for dim in [10,20,50]:
        pred=MiniBatchKMeans(n_clusters=chosen_k,n_init=10,random_state=seed,batch_size=2048).fit_predict(x[:,:dim]); sensitivity.append({'variant':f'PCA_{dim}','ari_vs_consensus':adjusted_rand_score(labels,pred)})
    for scaler in ['standard','robust','minmax']:
        mat=sparse.load_npz(processed/f'X_full_{scaler}.npz'); pred=MiniBatchKMeans(n_clusters=chosen_k,n_init=10,random_state=seed,batch_size=2048).fit_predict(mat); sensitivity.append({'variant':f'scaler_{scaler}','ari_vs_consensus':adjusted_rand_score(labels,pred)})
    pd.DataFrame(sensitivity).to_csv(tables/'phase3_sensitivity.csv',index=False,encoding='utf-8-sig')

    half=len(x)//2; drift=[]
    for j in range(10):
        a=np.asarray(x[:half,j]); b=np.asarray(x[half:,j]); ks=ks_2samp(a,b)
        drift.append({'feature':f'PC{j+1}','psi':_psi(a,b),'ks_statistic':ks.statistic,'ks_pvalue':ks.pvalue,'alert':bool(_psi(a,b)>.2 or ks.statistic>.1)})
    pd.DataFrame(drift).to_csv(tables/'phase3_drift_baseline.csv',index=False,encoding='utf-8-sig')
    bonus=run_bonus_analysis(config,x_pca=x,consensus_labels=labels,consensus_sample_labels=y_sample,candidate_names=names,candidate_labelings=labelings,sample_indices=idx)
    registry={'created_at_utc':utc_now_iso(),'model':'consensus_random_forest_assigner','consensus_k':chosen_k,'validation_accuracy':validation_accuracy,'explanation_tree_validation_accuracy':tree_validation_accuracy,'explanation_tree_training_accuracy':tree_training_accuracy,'shap_status':shap_status,'assigner_sha256':sha256_file(models/'phase3_consensus_assigner.joblib'),'rows':len(x),'features':x.shape[1],'centroids':list(centroids.shape),'bonus_points_targeted':bonus['points_targeted']}
    write_json(registry,artifacts/'model_registry.json')
    manifest={'phase':3,'status':'completed','elapsed_seconds':time.perf_counter()-started,'consensus_k':chosen_k,'candidate_algorithms':names,'validation_accuracy':validation_accuracy,'explanation_tree_validation_accuracy':tree_validation_accuracy,'explanation_tree_training_accuracy':tree_training_accuracy,'shap_status':shap_status,'cluster_sizes':{str(k):int(v) for k,v in zip(*np.unique(labels,return_counts=True))},'bonus_status':bonus['status'],'bonus_points_targeted':bonus['points_targeted']}
    write_json(manifest,artifacts/'phase3_run_summary.json'); return manifest
