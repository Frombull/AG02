AG2 - Engenharia de Software (GES)
Marco Di Toro - matricula 150

Classifica o canal de vendas (HoReCa ou Retail) do dataset Wholesale customers usando uma Decision Tree.

Como rodar:

pip install -r requirements.txt
python main.py

O script le o wholesale.csv, converte Channel e Region para inteiros, reordena as colunas, separa 80% treino e 20% teste, treina o modelo, mostra o classification_report e depois permite digitar uma amostra para classificar.
