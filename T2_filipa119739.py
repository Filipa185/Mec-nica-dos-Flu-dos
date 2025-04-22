import numpy as np
from scipy.optimize import differential_evolution, Bounds, fsolve
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter("ignore", RuntimeWarning)

# Configurações do sistema (copiado da Tarefa 4.1)
CONFIG = {
    'DEPÓSITO': {
        'nível_mín': 2.0,
        'nível_máx': 7.0,
        'nível_inicial': 4.0,
        'area': 185.0
    },
    'BOMBA': {
        'eficiência': 0.65,
        'altura_máx': 260,
        'curva_a': 0.002
    },
    'TUBULAÇÕES': {
        'diâmetro': 0.3,
        'atrito': 0.02,
        'comprimento_RF': 2500,
        'comprimento_PR': 5000
    },
    'TEMPO': {
        'horizonte': 24,
        'passo': 0.25
    }
}

# Tarifário energético ajustado (copiado da Tarefa 4.1)
TARIFAS = [
    (0, 0.065), (3, 0.055), (6, 0.075), (9, 0.085),
    (12, 0.095), (15, 0.105), (18, 0.095), (21, 0.075), (24, 0.065)
]

# Funções auxiliares (copiadas da Tarefa 4.1)
def calcular_tarifa(t):
    """Retorna o custo da energia para um dado tempo t"""
    for i in range(len(TARIFAS)-1):
        if TARIFAS[i][0] <= t < TARIFAS[i+1][0]:
            return TARIFAS[i][1]
    return TARIFAS[-1][1]

def demanda_vc_max(t):
    """Demanda máxima da região VC"""
    return (-1.19333e-7*t**7 - 4.90754e-5*t**6 + 3.733e-3*t**5 - 0.09621*t**4 + 
            1.03965*t**3 - 3.8645*t**2 - 1.0124*t + 75.393)

def demanda_vc_min(t):
    """Demanda mínima da região VC"""
    return (1.19333e-7*t**7 - 6.54846e-5*t**6 + 4.1432e-3*t**5 - 0.100585*t**4 + 
            1.05577*t**3 - 3.85966*t**2 - 1.32657*t + 75.393)

def demanda_residencial(t):
    """Demanda da região R"""
    return -0.004*t**3 + 0.09*t**2 + 0.1335*t + 20

def calcular_vazao_bomba(nivel_deposito):
    """Calcula a vazão da bomba para um dado nível do depósito"""
    def equacao(Q):
        perda_PR = (32 * CONFIG['TUBULAÇÕES']['atrito'] * CONFIG['TUBULAÇÕES']['comprimento_PR'] * (Q/3600)**2) / \
                  (CONFIG['TUBULAÇÕES']['diâmetro']**5 * 9.81 * np.pi**2)
        
        perda_RF = (32 * CONFIG['TUBULAÇÕES']['atrito'] * CONFIG['TUBULAÇÕES']['comprimento_RF'] * (Q/3600)**2) / \
                  (CONFIG['TUBULAÇÕES']['diâmetro']**5 * 9.81 * np.pi**2)
        
        H = CONFIG['BOMBA']['altura_máx'] - CONFIG['BOMBA']['curva_a'] * Q**2
        return H - (perda_PR + perda_RF + nivel_deposito)
    
    try:
        Q = fsolve(equacao, 150)[0]
        return max(0, Q)
    except:
        return 0

def formatar_horario(hora):
    """Formata horas decimais para formato HH:MM"""
    horas = int(hora)
    minutos = int(round((hora - horas) * 60))
    if minutos >= 60:
        horas += 1
        minutos -= 60
    return f"{horas:02d}:{minutos:02d}"

# Classe OtimizadorBomba (copiada da Tarefa 4.1)
class OtimizadorBomba:
    def __init__(self, modo='max', n_ciclos=3):
        self.modo = modo
        self.n_ciclos = n_ciclos
        self.tempo = np.arange(0, CONFIG['TEMPO']['horizonte'], CONFIG['TEMPO']['passo'])
    
    def simular(self, horarios, duracoes):
        """Simula o sistema para dados horários e durações de ativação da bomba"""
        try:
            # Arredonda e aplica restrições
            horarios = np.round(np.array(horarios) / 0.25) * 0.25
            duracoes = np.round(np.array(duracoes) / 0.25) * 0.25
            duracoes = np.clip(duracoes, 0.5, 4.0)  # Duração entre 30min e 4h
            
            # Verifica horários dentro de 24h
            if any(h + d > 24 for h, d in zip(horarios, duracoes)):
                return None
                
            # Verifica intervalo mínimo de 1h entre ativações
            horarios_ordenados = sorted(zip(horarios, duracoes), key=lambda x: x[0])
            for i in range(1, len(horarios_ordenados)):
                if horarios_ordenados[i][0] - (horarios_ordenados[i-1][0] + horarios_ordenados[i-1][1]) < 1.0:
                    return None
            
            bomba = np.zeros_like(self.tempo)
            for h, d in zip(horarios, duracoes):
                bomba += ((self.tempo >= h) & (self.tempo < h + d)).astype(float)
            
            nivel = np.zeros_like(self.tempo)
            potencia = np.zeros_like(self.tempo)
            custo = np.zeros_like(self.tempo)
            nivel[0] = CONFIG['DEPÓSITO']['nível_inicial']
            
            for i in range(1, len(self.tempo)):
                t = self.tempo[i]
                
                if self.modo == 'max':
                    demanda_total = demanda_vc_max(t) + demanda_residencial(t)
                else:
                    demanda_total = demanda_vc_min(t) + demanda_residencial(t)
                
                if bomba[i] > 0.5:
                    Q_bomba = calcular_vazao_bomba(nivel[i-1])
                    Q_direto = min(Q_bomba, demanda_residencial(t))
                    Q_tanque = Q_bomba - Q_direto
                    
                    H = CONFIG['BOMBA']['altura_máx'] - CONFIG['BOMBA']['curva_a'] * Q_bomba**2
                    potencia[i] = (1000 * 9.81 * (Q_bomba/3600) * H) / (1000 * CONFIG['BOMBA']['eficiência'])
                else:
                    Q_bomba, Q_direto, Q_tanque, potencia[i] = 0, 0, 0, 0
                
                Q_retirada = max(0, demanda_total - Q_direto)
                delta_nivel = (Q_tanque - Q_retirada) * CONFIG['TEMPO']['passo'] / CONFIG['DEPÓSITO']['area']
                nivel[i] = nivel[i-1] + delta_nivel
                
                if nivel[i] < CONFIG['DEPÓSITO']['nível_mín'] or nivel[i] > CONFIG['DEPÓSITO']['nível_máx']:
                    return None
                
                custo[i] = custo[i-1] + potencia[i] * calcular_tarifa(t) * CONFIG['TEMPO']['passo']
            
            bomba_schedule = list(zip(horarios, duracoes))
            
            return {
                'tempo': self.tempo,
                'nivel': nivel,
                'potencia': potencia,
                'custo': custo,
                'bomba': bomba,
                'modo': self.modo,
                'custo_total': custo[-1],
                'bomba_schedule': bomba_schedule,
                'nivel_final': nivel[-1]
            }
        except:
            return None
    
    def funcao_objetivo(self, x):
        sim = self.simular(x[:self.n_ciclos], x[self.n_ciclos:])
        if sim is None:
            return float('inf')
        
        custo_total = sim['custo_total']
        nivel_min = np.min(sim['nivel'])
        nivel_max = np.max(sim['nivel'])
        nivel_final = sim['nivel_final']
        
        # Penalizações ajustadas
        if nivel_min < CONFIG['DEPÓSITO']['nível_mín']:
            custo_total += 500 * (CONFIG['DEPÓSITO']['nível_mín'] - nivel_min)**2
        if nivel_max > CONFIG['DEPÓSITO']['nível_máx']:
            custo_total += 500 * (nivel_max - CONFIG['DEPÓSITO']['nível_máx'])**2
        
        # Penalização suave para nível final
        custo_total += 200 * abs(nivel_final - CONFIG['DEPÓSITO']['nível_inicial'])
        
        # Penalização adicional para custos altos
        if custo_total > 70:
            custo_total += (custo_total - 70)**2
            
        return custo_total
    
    def otimizar(self):
        passo = CONFIG['TEMPO']['passo']
        bounds = Bounds(
            [0]*self.n_ciclos + [0.5]*self.n_ciclos,
            [24-passo]*self.n_ciclos + [4]*self.n_ciclos
        )
        
        try:
            resultado = differential_evolution(
                self.funcao_objetivo,
                bounds=list(zip(bounds.lb, bounds.ub)),
                maxiter=1000,
                popsize=50,
                tol=0.0001,
                polish=True,
                seed=42,
                strategy='best1bin',
                recombination=0.9,
                mutation=(0.5, 1.0)
            )
            
            if resultado.success:
                return self.simular(resultado.x[:self.n_ciclos], resultado.x[self.n_ciclos:])
        except:
            pass
        
        return None

def plotar_resultados(resultados):
    if resultados is None:
        print("❌ Nenhum resultado válido para mostrar.")
        return
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Estado da bomba
    axs[0].step(resultados['tempo'], resultados['bomba'], where='post', color='blue')
    axs[0].set_ylabel('Estado da Bomba')
    axs[0].set_yticks([0, 1])
    axs[0].grid(True)
    
    # Nível do depósito
    axs[1].plot(resultados['tempo'], resultados['nivel'], color='green')
    axs[1].axhline(CONFIG['DEPÓSITO']['nível_mín'], color='r', linestyle='--', label='Limite mínimo')
    axs[1].axhline(CONFIG['DEPÓSITO']['nível_máx'], color='r', linestyle='--', label='Limite máximo')
    axs[1].axhline(CONFIG['DEPÓSITO']['nível_inicial'], color='b', linestyle=':', label='Nível inicial/final')
    axs[1].set_ylabel('Nível do Depósito (m)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Potência da bomba
    axs[2].plot(resultados['tempo'], resultados['potencia'], color='orange')
    axs[2].set_ylabel('Potência (kW)')
    axs[2].grid(True)
    
    # Custo acumulado
    axs[3].plot(resultados['tempo'], resultados['custo'], color='purple')
    axs[3].set_ylabel('Custo Acumulado (€)')
    axs[3].set_xlabel('Tempo (horas)')
    axs[3].grid(True)
    
    plt.suptitle(f"Resultados - Cenário {resultados['modo'].upper()} (Custo Total: €{resultados['custo_total']:.2f}, Nível Final: {resultados['nivel_final']:.2f}m)")
    plt.tight_layout()
    plt.savefig(f"resultados_{resultados['modo']}.png", dpi=300)
    plt.show()

def executar_otimizacao():
    resultados_finais = {}
    
    for modo in ['max', 'min']:
        print(f"\n=== OTIMIZANDO CENÁRIO {modo.upper()} ===")
        
        melhor_resultado = None
        melhor_custo = float('inf')
        
        for n_ciclos in range(2, 5):  # Testa 2, 3 e 4 ciclos
            print(f"\nTentando com {n_ciclos} ciclo(s) de bombeamento...")
            
            otimizador = OtimizadorBomba(modo=modo, n_ciclos=n_ciclos)
            resultado = otimizador.otimizar()
            
            if resultado and resultado['custo_total'] < melhor_custo:
                melhor_resultado = resultado
                melhor_custo = resultado['custo_total']
                print(f"✔ Melhor solução encontrada: Custo = €{melhor_custo:.2f}")
                print(f"   Nível final: {melhor_resultado['nivel_final']:.2f}m")
            else:
                print("✖ Nenhuma solução válida encontrada ou pior que a atual.")
        
        if melhor_resultado:
            resultados_finais[modo] = melhor_resultado
            print(f"\n⭐ Melhor resultado para cenário {modo.upper()}:")
            print(f" - Custo total: €{melhor_resultado['custo_total']:.2f}")
            print(f" - Nível final: {melhor_resultado['nivel_final']:.2f}m")
            print(" - Horários de ativação:")
            for inicio, duracao in melhor_resultado['bomba_schedule']:
                print(f"   - Início: {formatar_horario(inicio)}, Duração: {formatar_horario(duracao)}")
            plotar_resultados(melhor_resultado)
        else:
            print(f"\n⚠ Não foi possível encontrar uma solução válida para o cenário {modo.upper()}")
    
    return resultados_finais

# Classe OtimizadorRobusto e funções relacionadas (Tarefa 4.2)
class OtimizadorRobusto:
    def __init__(self, penalizacao=0.10):
        self.penalizacao = penalizacao
        self.tempo = np.arange(0, CONFIG['TEMPO']['horizonte'], CONFIG['TEMPO']['passo'])
        self.resultados_max = None
        self.resultados_min = None
    
    def carregar_cenarios(self, resultados_max, resultados_min):
        """Carrega os resultados ótimos dos cenários extremos"""
        self.resultados_max = resultados_max
        self.resultados_min = resultados_min
    
    def calcular_sobreposicao(self):
        """Calcula a sobreposição dos horários de bombeamento entre cenários"""
        if not self.resultados_max or not self.resultados_min:
            return None
            
        bomba_max = self.resultados_max['bomba']
        bomba_min = self.resultados_min['bomba']
        sobreposicao = np.minimum(bomba_max, bomba_min)
        
        return {
            'tempo': self.tempo,
            'sobreposicao': sobreposicao,
            'bomba_max': bomba_max,
            'bomba_min': bomba_min
        }
    
    def simular_robusto(self, x):
        """Simula um cenário robusto com penalizações"""
        horarios = x[:3]
        duracoes = x[3:6]
        fator_ajuste = x[6]  # Fator para ajustar entre cenários (0-1)
        
        # Aplica restrições
        horarios = np.round(np.array(horarios) / 0.25) * 0.25
        duracoes = np.round(np.array(duracoes) / 0.25) * 0.25
        duracoes = np.clip(duracoes, 0.5, 4.0)
        
        # Verifica horários
        if any(h + d > 24 for h, d in zip(horarios, duracoes)):
            return None
            
        # Cria perfil de bombeamento
        bomba = np.zeros_like(self.tempo)
        for h, d in zip(horarios, duracoes):
            bomba += ((self.tempo >= h) & (self.tempo < h + d)).astype(float)
        
        # Simula ambos os cenários com o mesmo perfil de bombeamento
        sim_max = self._simular_cenario(bomba, 'max')
        sim_min = self._simular_cenario(bomba, 'min')
        
        if not sim_max or not sim_min:
            return None
            
        # Calcula custos totais com penalizações
        custo_max = sim_max['custo_total'] + sim_max['penalizacao_total']
        custo_min = sim_min['custo_total'] + sim_min['penalizacao_total']
        
        # Combina os custos conforme fator de ajuste
        custo_total = fator_ajuste * custo_max + (1 - fator_ajuste) * custo_min
        
        return {
            'tempo': self.tempo,
            'bomba': bomba,
            'nivel_max': sim_max['nivel'],
            'nivel_min': sim_min['nivel'],
            'custo_max': sim_max['custo'],
            'custo_min': sim_min['custo'],
            'penalizacao_max': sim_max['penalizacao_total'],
            'penalizacao_min': sim_min['penalizacao_total'],
            'custo_total': custo_total,
            'horarios': list(zip(horarios, duracoes)),
            'fator_ajuste': fator_ajuste
        }
    
    def _simular_cenario(self, bomba, modo):
        """Simula um cenário individual com penalizações"""
        nivel = np.zeros_like(self.tempo)
        potencia = np.zeros_like(self.tempo)
        custo = np.zeros_like(self.tempo)
        penalizacao = np.zeros_like(self.tempo)
        nivel[0] = CONFIG['DEPÓSITO']['nível_inicial']
        violacao_continua = 0
        
        for i in range(1, len(self.tempo)):
            t = self.tempo[i]
            
            if modo == 'max':
                demanda_total = demanda_vc_max(t) + demanda_residencial(t)
            else:
                demanda_total = demanda_vc_min(t) + demanda_residencial(t)
            
            if bomba[i] > 0.5:
                Q_bomba = calcular_vazao_bomba(nivel[i-1])
                Q_direto = min(Q_bomba, demanda_residencial(t))
                Q_tanque = Q_bomba - Q_direto
                
                H = CONFIG['BOMBA']['altura_máx'] - CONFIG['BOMBA']['curva_a'] * Q_bomba**2
                potencia[i] = (1000 * 9.81 * (Q_bomba/3600) * H) / (1000 * CONFIG['BOMBA']['eficiência'])
            else:
                Q_bomba, Q_direto, Q_tanque, potencia[i] = 0, 0, 0, 0
            
            Q_retirada = max(0, demanda_total - Q_direto)
            delta_nivel = (Q_tanque - Q_retirada) * CONFIG['TEMPO']['passo'] / CONFIG['DEPÓSITO']['area']
            nivel[i] = nivel[i-1] + delta_nivel
            
            # Verifica violações e aplica penalizações progressivas
            if nivel[i] < CONFIG['DEPÓSITO']['nível_mín'] or nivel[i] > CONFIG['DEPÓSITO']['nível_máx']:
                violacao_continua += 1
                penalizacao[i] = self.penalizacao * violacao_continua
            else:
                violacao_continua = 0
            
            custo[i] = custo[i-1] + potencia[i] * calcular_tarifa(t) * CONFIG['TEMPO']['passo'] + penalizacao[i]
        
        return {
            'nivel': nivel,
            'potencia': potencia,
            'custo': custo,
            'penalizacao_total': np.sum(penalizacao),
            'custo_total': custo[-1]
        }
    
    def funcao_objetivo(self, x):
        sim = self.simular_robusto(x)
        if sim is None:
            return float('inf')
        
        # Ponderamos mais o cenário crítico (MAX) com peso 0.7
        custo_ponderado = 0.7 * (sim['custo_max'][-1] + sim['penalizacao_max']) + \
                         0.3 * (sim['custo_min'][-1] + sim['penalizacao_min'])
        
        # Penalização adicional se ambos os cenários tiverem violações significativas
        if sim['penalizacao_max'] > 5 or sim['penalizacao_min'] > 5:
            custo_ponderado += 10
            
        return custo_ponderado
    
    def otimizar(self):
        bounds = Bounds(
            [0]*3 + [0.5]*3 + [0],  # 3 horários, 3 durações, 1 fator
            [24]*3 + [4]*3 + [1]
        )
        
        resultado = differential_evolution(
            self.funcao_objetivo,
            bounds=list(zip(bounds.lb, bounds.ub)),
            maxiter=500,
            popsize=30,
            tol=0.001,
            seed=42
        )
        
        if resultado.success:
            return self.simular_robusto(resultado.x)
        return None

def plotar_resultados_robustos(resultado):
    if not resultado:
        print("Nenhum resultado válido para mostrar")
        return
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Estado da bomba
    axs[0].step(resultado['tempo'], resultado['bomba'], where='post', color='blue')
    axs[0].set_ylabel('Estado da Bomba')
    axs[0].set_yticks([0, 1])
    axs[0].grid(True)
    
    # Níveis dos depósitos
    axs[1].plot(resultado['tempo'], resultado['nivel_max'], 'r-', label='Cenário MAX')
    axs[1].plot(resultado['tempo'], resultado['nivel_min'], 'b-', label='Cenário MIN')
    axs[1].axhline(CONFIG['DEPÓSITO']['nível_mín'], color='r', linestyle='--')
    axs[1].axhline(CONFIG['DEPÓSITO']['nível_máx'], color='r', linestyle='--')
    axs[1].set_ylabel('Nível (m)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Custos
    axs[2].plot(resultado['tempo'], resultado['custo_max'], 'r-', label='Custo MAX')
    axs[2].plot(resultado['tempo'], resultado['custo_min'], 'b-', label='Custo MIN')
    axs[2].set_ylabel('Custo (€)')
    axs[2].legend()
    axs[2].grid(True)
    
    # Penalizações
    axs[3].bar(resultado['tempo'], resultado['penalizacao_max'], color='r', alpha=0.5, label='Penal. MAX')
    axs[3].bar(resultado['tempo'], resultado['penalizacao_min'], color='b', alpha=0.5, label='Penal. MIN')
    axs[3].set_ylabel('Penalização (€)')
    axs[3].set_xlabel('Tempo (horas)')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.suptitle(f"Operação Robusta (Custo Total: €{resultado['custo_total']:.2f}, Fator: {resultado['fator_ajuste']:.2f})")
    plt.tight_layout()
    plt.savefig("resultado_robusto.png", dpi=300)
    plt.show()

def executar_tarefa_4_2(resultados_tarefa_4_1):
    print("\n=== TAREFA 4.2 - OPERAÇÃO ROBUSTA ===")
    print(f"Penalização configurada: €{0.10}/hora (acumulativa)")
    
    otimizador = OtimizadorRobusto(penalizacao=0.10)
    otimizador.carregar_cenarios(resultados_tarefa_4_1.get('max'), resultados_tarefa_4_1.get('min'))
    
    # Calcula sobreposição entre cenários
    sobreposicao = otimizador.calcular_sobreposicao()
    if sobreposicao:
        plt.figure(figsize=(10, 4))
        plt.step(sobreposicao['tempo'], sobreposicao['bomba_max'], 'r-', where='post', label='Cenário MAX')
        plt.step(sobreposicao['tempo'], sobreposicao['bomba_min'], 'b-', where='post', label='Cenário MIN')
        plt.step(sobreposicao['tempo'], sobreposicao['sobreposicao'], 'g-', where='post', linewidth=2, label='Sobreposição')
        plt.ylabel('Estado da Bomba')
        plt.xlabel('Tempo (horas)')
        plt.legend()
        plt.grid(True)
        plt.title('Sobreposição de Horários de Bombeamento')
        plt.tight_layout()
        plt.savefig("sobreposicao_bombeamento.png", dpi=300)
        plt.show()
    
    # Otimização robusta
    print("\nOtimizando operação robusta...")
    resultado_robusto = otimizador.otimizar()
    
    if resultado_robusto:
        print("\n⭐ Resultado da Operação Robusta:")
        print(f"- Custo total ponderado: €{resultado_robusto['custo_total']:.2f}")
        print(f"- Fator de ajuste: {resultado_robusto['fator_ajuste']:.2f} (0=MIN, 1=MAX)")
        print(f"- Penalização MAX: €{resultado_robusto['penalizacao_max']:.2f}")
        print(f"- Penalização MIN: €{resultado_robusto['penalizacao_min']:.2f}")
        print("- Horários de bombeamento:")
        for inicio, duracao in resultado_robusto['horarios']:
            print(f"  {formatar_horario(inicio)} - {formatar_horario(inicio+duracao)} ({formatar_horario(duracao)})")
        
        plotar_resultados_robustos(resultado_robusto)
    else:
        print("✖ Não foi possível encontrar uma solução robusta válida.")
    
    return resultado_robusto

# Execução principal
if __name__ == "__main__":
    # Primeiro executamos a Tarefa 4.1 para obter os cenários base
    resultados_tarefa_4_1 = executar_otimizacao()
    
    # Em seguida executamos a Tarefa 4.2
    if resultados_tarefa_4_1:
        executar_tarefa_4_2(resultados_tarefa_4_1)