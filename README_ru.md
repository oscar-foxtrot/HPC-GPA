# Практическое задание по курсу "Суперкомпьютеры и параллельная обработка данных" (СКиПОД)

- [Switch to English](README.md)  
- Русский (выбран)

Описать алгоритм и реализовать его с использованием суперкомпьютерных мощностей [МГУ].
В данной работе рассматривается алгоритм <b>Generalized Procrustes Analysis (GPA)</b>.

[Ссылка на страницу с описанием алгоритма](https://algowiki-project.org/ru/%D0%A3%D1%87%D0%B0%D1%81%D1%82%D0%BD%D0%B8%D0%BA:OscarFox/%D0%9E%D0%B1%D0%BE%D0%B1%D1%89%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%BF%D1%80%D0%BE%D0%BA%D1%80%D1%83%D1%81%D1%82%D0%BE%D0%B2_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7)

Использован суперкомпьютер "Ломоносов-2". Описание среды:

использованные модули:
  1) xalt   2) slurm/15.08.1   3) intel/2019.5   4) mkl/2019.2   5) gcc/9.1   6) openmpi/4.0.1-icc

Опции компиляции для gpa_parallel.cpp:

mpicxx gpa_parallel.cpp -I"eigen-3.4.1/" -std=c++17 -w -fopenmp -o executable_file
