## Generates latex code for table creation from regression data ##

'''
\begin{table}[ht]
\footnotesize
\centering
\captionsetup{justification=centering}
\caption[]{
	\textsc{Estimativa dos ParÃ¢metros por GMM}
	\begin{minipage}{\linewidth}
	  \hspace*{-1.5cm}
	  \vbox{
	  \begin{flalign*}
		  & E_t\Delta\ln(C(t+1))=const.+sE_t[i(t+1)-\pi(t+1)]+ \\
		  & hE_t\Delta\ln(C(t))+lE_t\Delta\ln(L(t+1))+bE_t\Delta\ln(Y(t+1))
          \end{flalign*}
	  }
        \end{minipage}
}
\label{table:tabgmm} 
\vspace*{-0.5cm}
\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}
\begin{tabular}{l*{2}{c}}
\hline\hline\\[-1.8ex] 
	& (1) & (2) \\ 
	Coeficiente & Conj. Instrumentos 1 & Conj. Instrumentos 2 \\
\hline\\[-1.8ex] 
\quad\textit{s} & 5.928 & -8.216 \\
  & (27.545) & (22.339) \\
[1em]
\quad\textit{h} & -0.245 & -0.228 \\
  & (0.434) & (0.402) \\
[1em]
\quad\textit{l} & 0.122 & -0.035 \\
  & (0.349) & (0.304) \\
[1em]
\quad\textit{b} & 0.336 & 0.841\sym{**} \\
  & (0.479) & (0.368)            \\
\hline\\[-1.8ex] 
	\textit{EstatÃ­stica J} & 2.1434 & 6.8354 \\
	\textit{p-valor} & 0.7094 & 0.3363  \\
\hline\hline\\[-1.8ex] 
\multicolumn{3}{l}{\footnotesize Desvios-padrÃ£o em parÃªnteses.}\\
\multicolumn{3}{l}{\footnotesize \sym{*} \(p<0.1\), \sym{**} \(p<0.05\), \sym{***} \(p<0.01\)}\\
\end{tabular}
\end{table}
'''
