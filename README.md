# Implementation_ML


## Files

| No | Title             | Description |
|----|-------------------|-------------|
| 1  | Linear Regression |             |




## 開發規範

### 環境管理
- **創建環境** 初次部署時需要輸入此指令
  - mac -> `conda create -n kbs python=3.8.5 anaconda`
  - windows -> `conda create -n kbs python=3.8.5 anaconda`  
  - linux&server -> `sudo /opt/conda/bin/conda create -n kbs python=3.8.5 anaconda`
- **克隆工具**
  - dstools
    - `git clone git@github.com:aaatechTeam/dstools.git`
- **啟動虛擬環境** 每次開發或部署時皆須輸入此指令
  - mac -> `conda activate kbs`
  - windows -> `activate kbs`
  - linux -> `source activate kbs`
  - linux&server -> `source /opt/conda/bin/activate kbs`
- **離開虛擬環境** 
  - mac -> `conda deactivate`
  - windows -> `deactivate`
  - linux -> `source /opt/conda/bin/deactivate`  
  - linux&server -> `source /opt/conda/bin/deactivate`  
- **啟動專案**
  - `(kbs) python3 run_server.py`
- **安裝套件** 初次部署或有新的套件更新時需要輸入此指令
  - `(kbs) pip install -r requirements.txt`
  - `(kbs) pip install -r requirements.txt --user` **if need auth**
- **輸出套件** 有新的套件更新至環境時需要輸入此指令
  - `(kbs) pip freeze > requirements.txt`

### packages Memo

- Pandas In Jupyter Use
  - [tqdm](https://github.com/tqdm/tqdm)
  - [qgrid](https://github.com/quantopian/qgrid)
  - [swifter](https://github.com/jmcarpenter2/swifter)
  - [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)
- Visulisation
  - [seaborn](https://seaborn.pydata.org/)
  - [chartify](https://github.com/spotify/chartify)
- Others
  - [feather-format](https://pypi.org/project/feather-format/)