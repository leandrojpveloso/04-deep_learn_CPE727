# 1: Introduction

The code inside this folder, `1-import`, is destinate to read the raw files and harmonize the files. It means, adjust names, labels, formats and minor cleannings.
The main idea is that we have a set of data in dta/rds format that we can easy merge and append to create the data we requires for our astudy.

# 2: Source data

Federal public procurement in Brazil has three main source, two of them is public and one is restrict.

## 2.1 Procurement data

* [Portal da transparência](https://www.portaltransparencia.gov.br/origem-dos-dados): A website that we have access to the 
federal expense composition. It allows to download data of tenders and contracts monthly. 
* [Compras dados](http://compras.dados.gov.br/): It is a website that allows to download almost every piece of information in the tender procces. However,
it is quite more difficult to download the data.
* SIASG datawarehouse: this is the original source that has restrict access.

##  2.2 Product Classification 
The procurement processes dataset and the contracts dataset contain detailed information on purchased items. Items are classified using the CATSER (catálogo de serviços) catalog for service, and the CATMAT (catálogo de materiais) catalog for goods.  Using these classifications, it is possible categorizes items in some agreagation level. 

Download option:
* [CATMAT/CATSER](https://www.gov.br/compras/pt-br/acesso-a-informacao/consulta-detalhada/planilha-catmat-catser): www.gov.br
* [CATMAT](http://compras.dados.gov.br/docs/lista-metodos-materiais.html): compras.dados.gov
* [CATSER](http://compras.dados.gov.br/docs/lista-metodos-servicos.html): compras.dados.gov

### [CATMAT](https://www.gov.br/saude/pt-br/acesso-a-informacao/gestao-do-sus/economia-da-saude/banco-de-precos-em-saude/catalogo-de-materiais-2013-catmat)
  Goods are detailed up to 6 digits levels. The materials are classifications that are aggregable by 2, 3, and 5 digits level.
https://www.gov.br/compras/pt-br/acesso-a-informacao/consulta-detalhada/planilha-catmat-catser/view 

### [CATSER](http://compras.dados.gov.br/): 
  Services are detailed up to 5 digits levels. The materials are classifications that are aggregable by 1, 2, 3, and 4 digits level.
 
# 3: Programs

The import proccess were made in the KCP project and FA project.=