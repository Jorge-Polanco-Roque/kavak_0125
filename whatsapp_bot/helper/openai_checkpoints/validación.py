import pandas as pd

df = pd.read_csv(
    "/Users/A1064331/Desktop/pruebas/Kavak/test_1/input/sample_caso_ai_engineer.csv",  # Ruta a tu CSV
    delimiter=",",
    # Si no tiene encabezados en el CSV, podrías usar 'names=[...]'
)

#print(df.head(2))

# ¿Cuál es el precio promedio de los autos con bluetooth habilitado?
print(df[(df['bluetooth']=='Sí')]['price'].mean()) #328297.96907216497