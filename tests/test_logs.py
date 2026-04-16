import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import cv2
from proctor.engine import ProctorEngine

def main():
    # Inicializa o motor (ele já configura o logger e o gaze internamente)
    engine = ProctorEngine()
    cap = cv2.VideoCapture(0)

    print("=== TESTE DE LOGS ATIVO ===")
    print("1. Olhe para o lado por 3s para gerar um bloqueio.")
    print("2. Pressione 'q' para encerrar e salvar os logs.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # O motor processa o frame e decide se deve logar algo
        state = engine.update(frame)
        
        # Exibe o estado atual na tela para você acompanhar
        cv2.putText(frame, f"ESTADO: {state}", (20, 50), 1, 2, (0, 255, 0), 2)
        cv2.imshow("Teste de Engine e Logs", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Teste finalizado. Verifique o arquivo 'events.jsonl'.")

if __name__ == "__main__":
    main()