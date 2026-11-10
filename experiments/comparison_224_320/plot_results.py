import csv, json
from pathlib import Path
from statistics import mean, stdev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

root = Path(__file__).resolve().parent
records = json.loads((root / 'raw_records.json').read_text(encoding='utf-8-sig'))
assert len(records) == 24
for record in records:
    args = yaml.safe_load(record['args'])
    assert args['seed'] == record['seed']
    assert args['window_size'] == ([7]*4 if record['configuration'] == '224' else [8,8,5,5])
    assert args['input_size'] == (None if record['configuration'] == '224' and record['phase'] == 1 else [3,int(record['configuration']),int(record['configuration'])])
    assert args['inter_window_mamba_stages'] == [0]
with (root / 'per_seed.csv').open('w', encoding='utf-8', newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['phase','dataset','seed','accuracy_224_percent','accuracy_320_percent','difference_percentage_points'])
    rows=[]
    for phase, dataset in enumerate(('MangoLeafBD','Combined','VN-MangoLeaf'), 1):
        differences=[]
        for seed in (42,123,2024,3333):
            values=[100*next(r for r in records if r['phase']==phase and r['seed']==seed and r['configuration']==c)['result']['test_metrics']['accuracy'] for c in ('224','320')]
            difference=values[1]-values[0]
            writer.writerow([phase,dataset,seed,*values,difference])
            differences.append(difference)
        rows.append((dataset,differences))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15})
fig, axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
for ax,(dataset, values) in zip(axes,rows):
    colors=['#9AA9BF' if x==0 else '#1D5FB8' if x>0 else '#D55E00' for x in values]
    ax.bar(['42','123','2024','3333'],values,color=colors,width=.65)
    ax.axhline(0,color='#64748b',linewidth=.8)
    ax.set_title(dataset,fontweight='bold',color='#0B2A5B')
    ax.set_xlabel('Seed')
    ax.text(.5,.96, f'Trung bình {mean(values):+.2f} ± {stdev(values):.2f} điểm %'.replace('.',','), transform=ax.transAxes,ha='center',va='top',fontsize=14)
    for i,v in enumerate(values):
        ax.annotate(f'{v:+.3f}'.replace('.',','),(i,v),xytext=(0,5 if v>=0 else -13),textcoords='offset points',ha='center',fontsize=13)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Chênh lệch Accuracy (điểm %)')
axes[0].set_ylim(-.35,2.5)
fig.suptitle('Test Accuracy: 320 × 320 trừ 224 × 224',fontsize=18,fontweight='bold',color='#0B2A5B')
fig.text(.5,.01,'Cùng kiến trúc có inter-window Mamba, bốn seed, cùng tập kiểm tra theo từng pha.',ha='center',fontsize=14)
fig.tight_layout(rect=(0,.07,1,.91))
fig.savefig(root/'accuracy_paired.png',dpi=220,bbox_inches='tight')
print('Validated 24 records and created chart')



