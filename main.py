# from ctransformers import AutoModelForCausalLM

# # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF", model_file="mistral-7b-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=50)

# print(llm("AI is going to"))


from ctransformers import AutoModelForCausalLM, AutoConfig, Config

conf = AutoConfig(
    Config(
        temperature=0.8,
        repetition_penalty=1.1,
        batch_size=52,
        max_new_tokens=1024,
        context_length=2048,
        gpu_layers=50,
    )
)
llm = AutoModelForCausalLM.from_pretrained(
    "./model/mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", config=conf
)

text = """
Resumen de Interrogatorio, Exploración Física y/o Estado Mental: 24/08/22 18:45 h Signos Vitales TAS TAD TAM FC FR TEMP SAT 02 PESO TALLA IMC INDICE DE CHOQUE 117 74 88 81 20 36.2 98 84.6 1.63 31.8 0.7 Femenino de 28 años de edad quien acude por referir salida de líquido trasvaginal de inico a las 18:00 h sin utilizar toalla sanitaria durante el trayecto, percibe disminución de movimientos fetales a razon de 1 movimiento cada hora, niega síntomas urinarios, niega síntomas respiratorios, niega datos de vasoespasmo, salida de tapón de moco con sangre. APNP: Gpo sang A Rh positivo Aplico COVID 1 dosis y TdPa. APP: Alergias, Quirúrgicos, Crónico degenerativos, traumáticos y transfusionales Interrogados y negados, Tabaquismo, alcoholismo y toxicomanías interrogado y negado. AGO: Menarca a los 14 años Ritmo 30x7 IVSA a los 14 años #PS 5 MPF preservativos PAP: nunca realizado Gesta: 6 G1 Aborto esponteneo 2013 G2 Parto 17/04/2014 Peso 3500 gr sin complicaciones. G3 Parto 22/08/2015 Peso 3200 gr sin complicaciones G4 Parto 11/07/2017 Peso 3000 gr sin complicaciones G5 Aborto espontaneo 2021. G6 FUM 14/11/2021 Control prenatal: 4 consulta, consumió hemátinicos y folatos a partir de 12 SDG, se realizó 2 prueba de VIH y VDRL no reactivo. A la EF Consciente, orientada, cooperadora, normocefala con adecuada coloración e hidratación de mucosas, cardiopulmonar sin compromiso actual, abdomen con útero gestante con AFU de 30 cm, con PUVI, cefálico, longitudinal dorso a la derecha , con FCF 150 lpm (doptone), al TV cérvix posterior, dilatación 2cm, borramiento 30%, cavidad eutermica, valsalva y tarnier negativo, guante explorador limpio (se muestra a la paciente), extremidades sin edema ROTS normales. Giordano negativo bilateral. FUM 14/11/2021  CON 40.3 SDG POR FUM. ULTRASONIDO 30/03/2022 CON EMBARAZO DE 17.4 SDG, TRASPOLADO AL DÍA DE HOY, CON EMBARAZO DE 38.4 SDG. ULTRASONIDO 05/08/2022 CON EMBARAZO DE 35.5 SDG, TRASPOLADO AL DÍA DE HOY, CON EMBARAZO DE 38.3 SDG PLACENTA CON INSERCIÓN CORPORAL POSTERIOR GRADO II DE GRANUMM. LIQUIDO AMNIÓTICO CON bolsa mayor 6.2 CM. PESO FETAL ESTIMADO 2758 g, CIR5CULAR SIMPLE DE CORDON AL CUELLO.
"""

# Diagnóstico: G6A2P3 Embarazo de 38.4 SDG por USG traspolado del segundo trimestre + Trabajo de parto fase latente.

text2 = """
APNP: Gpo sang A Rh positivo Aplico COVID 1 dosis y TdPa. APP: Alergias, Quirúrgicos, Crónico degenerativos, traumáticos y transfusionales Interrogados y negados, Tabaquismo, alcoholismo y toxicomanías interrogado y negado. AGO: Menarca a los 14 años Ritmo 30x7 IVSA a los 14 años #PS 5 MPF preservativos PAP: nunca realizado Gesta: 6 G1 Aborto esponteneo 2013 G2 Parto 17/04/2014 Peso 3500 gr sin complicaciones. G3 Parto 22/08/2015 Peso 3200 gr sin complicaciones G4 Parto 11/07/2017 Peso 3000 gr sin complicaciones G5 Aborto espontaneo 2021. G6 FUM 14/11/2021
"""

text3 = """El presidente de los Estados Unidos, Joe Biden, visitó la Ciudad de México el viernes"""



# # prompt = f"De la nota clinica: {text} Cuáles son las entidades clínicas que se pueden diagnosticar en esta paciente?"
# # prompt = f"Lista todas las entidades nombradas con su respectiva etiqueta en la siguiente oración: {text3}"

# # prompt = f"Eres un experto en medicia y obstetricia. El siguiente texto es el historial de la paciente: {text2} '''Cuáles son las entidades clínicas que se pueden identificar en esta nota clinica?'''"
# prompt = f"Eres un experto en medicia y obstetricia. El siguiente texto es el historial de la paciente: {text} '''Lista las entidades clínicas que se pueden identificar en esta nota clinica, las entidades deben ser tal y cual aparecen, sin parafrasear, en idioma español y agrega la posición de inicio y fin de cada entidad en la nota.''''"
# # prompt = f"Eres un experto en medicia y obstetricia. El siguiente texto es el historial de la paciente: {text} '''Lista únicamente las entidades clínicas relacionadas con las gestas, partos, abortos y fechas en que suceden en esta nota clinica sin parafrasear y en idioma español'''"


while True:
    question = input("$> Escribe tu prompt: ")
    # prompt = f"Eres un experto en medicia y obstetricia. El siguiente texto es el historial de la paciente: {text} '''{question}''''"
    # prompt = f"Eres un asistente experto en medicina. {text} '''{question}''''"
    prompt = f"Act as a doctor. {text} '''{question}''''"
    mistral_prompt = f"<s>[INST] {prompt} [/INST]"

    answer = llm(
        mistral_prompt, temperature=0.7, repetition_penalty=1.15, max_new_tokens=2048
    )

    print(answer)
    print(f'{"="*150}\n\n')
