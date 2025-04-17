#!/bin/bash

response=$(curl -s -X 'POST' \
  'http://127.0.0.1:8000/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "content": "Conforme o contrato de locação firmado entre as partes, o índice de reajuste pactuado é o IGP‑M (Índice Geral de Preços do Mercado), com data de reajuste prevista para 1º de janeiro de 2024. O contrato teve início em 15 de março de 2023 e terá vigência até 14 de março de 2026, perfazendo um prazo total de 36 meses de locação. O valor mensal do aluguel foi fixado em R$ 3.500,00 e, de acordo com o acordo firmado, haverá dobra no valor devido ao aluguel no mês de dezembro, resultando em R$ 7.000,00 naquele mês."
}')

echo "$response" | jq
