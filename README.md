
<h1 align="center">MADA</h1>
<h2 align="center">(Manufatura Aditiva por Deposição a Arco)</h2>

---

## Esse Repositório Está Em Desenvolvimento

## Descrição
Este projeto visa criar ferramentas para utilização eficiente do método de
planejamento de caminho [PIXEL](https://www.mdpi.com/2075-4701/11/3/498) para MADA em Python/Notebooks.

## Vocabulário
- **Geometria (Geometry)**: Refere-se a qualquer mesh que forma uma superfície hermética de uma peça tridimensional.
- **Superfície Hermética (watertight)**: A superfície do sólido precisa se comportar como uma “casca” lisa e contínua em 3D.
- **Camada (Layer)**: Conjunto de ilhas e/ou furos obtidos a partir da interseção da geometria com um plano.
- **Ilha | Furo**: São polígonos (convexos ou côncavos) orientados; quando um polígono está no sentido anti-horário, é uma ilha, e quando está no sentido horário, é um furo.
- **Grade(Grid): Conjunto de pontos(x,y) em um plano z com uma mesma distância D vertical e horizontal entre si.
- **Peça(Part): Mesh de uma peça para manufatura aditiva que pode ser dicretizada em Camadas.
- **Excursão(Tour): Um conjunto ordenado de índices que representa um percurso em uma grade, abrangendo todos os pontos disponíveis, numerados de 0 a N-1, onde N é o total de pontos na grade.
- **Melhor Excursão(Best Tour): A excursão que minimiza a distância total percorrida ao visitar todos os pontos da grade.