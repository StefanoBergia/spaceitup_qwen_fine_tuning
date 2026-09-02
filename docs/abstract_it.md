# Abstract (IT)

**Ragionare prima di muoversi: embodied reasoning a bordo per la navigazione autonoma di
rover planetari**

La navigazione autonoma dei rover planetari si basa tradizionalmente su approcci puramente
geometrici, che stimano la percorribilità del terreno senza interpretarne il contenuto
semantico. Proponiamo di riformulare la scelta del percorso come un problema di *embodied
reasoning*: la traiettoria non è prodotta direttamente dall'immagine, ma derivata da
un'analisi esplicita della scena condotta dal punto di vista del rover.

Introduciamo una tecnica di specializzazione con cui un vision-language model (VLM)
compatto apprende a generare, prima della traiettoria, una catena di inferenze ancorata
alla scena osservata: quali regioni sono percorribili, quali ostacoli interrompono la via,
quali porzioni di ambiente restano occluse dietro di essi e quali traiettorie sono
effettivamente realizzabili dalla piattaforma. Il ragionamento non costituisce una
spiegazione a posteriori della traiettoria, ma la computazione intermedia che la
determina. La supervisione è ottenuta da scene simulate di cui la geometria è nota in
forma esatta, il che rende le catene di inferenze verificabili anziché soltanto
plausibili, e la loro esplicitazione in linguaggio naturale rende il processo decisionale
ispezionabile da un operatore.

Valutiamo la tecnica sia nella generazione diretta del percorso sia nella selezione
dell'alternativa corretta tra più candidate. L'analisi individua nella stima della
visibilità il principale fattore discriminante tra le configurazioni valutate, ed è
precisamente la capacità su cui il ragionamento esplicito incide: inferire quali tratti
del percorso risultino occlusi richiede una rappresentazione della geometria della scena
che va oltre ciò che è direttamente osservabile. Il sistema opera interamente su un
calcolatore NVIDIA Jetson Orin NX da 16 GB, senza connettività verso stazioni remote, ed è
stato validato in ambiente reale sul rover Ardito del Politecnico di Torino, concesso in
uso per le campagne sperimentali.

---

## Note redazionali

- **Taglio:** il contributo presentato è l'*embodied reasoning come tecnica* — il
  ragionamento ancorato alla scena come computazione che determina la traiettoria, non
  come spiegazione a posteriori. L'esecuzione a bordo e la validazione su rover reale
  sono la prova che la tecnica regge fuori dal laboratorio, non il contributo in sé.
  Dati, addestramento incrementale, A*, LoRA, Habitat, Cosmos Reason 3, Qwen3.5-2B/0.8B:
  tutto rimandato al corpo del paper, fuori dall'abstract.
- **Titoli alternativi** (stesso taglio, tono decrescente):
  - *Ragionare prima di muoversi: embodied reasoning a bordo per la navigazione autonoma
    di rover planetari* (in uso)
  - *Embodied reasoning a bordo: un VLM compatto che ragiona sulla scena per scegliere
    dove andare*
  - *Vedere, ragionare, muoversi: embodied reasoning per la pianificazione di percorso su
    rover planetari*
- **Scritto come lavoro concluso**, senza "in corso" o "da sviluppare". Va allineato allo
  stato reale prima della submission: al momento la componente di ragionamento non è
  ancora addestrata, il porting su Jetson è in corso e la campagna sul rover Ardito non è
  ancora avvenuta.
- **Nessun numero**: ogni cifra di risultato è omessa finché gli esperimenti non sono
  chiusi. Fonti quando si inseriranno: `outputs/eval_habitat/comparison.md` e
  `outputs/eval_habitat_choice/comparison.md`; per la classificazione la baseline da
  citare è `chance_accepted_excluding_direct` (≈0,50), non il caso uniforme.
