from gerar_dados_vitimas import gerar_dataset_vitimas

def main():
    df = gerar_dataset_vitimas(
        n_vitimas=5000,
        media_idade=35,
        desvio_idade=10,
        tipo_acidente="uniforme",
        nivel_ruido=0.03,
        seed=21
    )
    print("\nPrimeiras linhas do dataset gerado:")
    print(df.head())



if __name__ == "__main__":
    main()