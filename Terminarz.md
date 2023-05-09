0.Ustanowienie zespołów projektowychSkład zespołów zgłosić do 14III.

# 1. do 20.III
Przygotowanie planszy i scenariusza przejazdu Wymagania:
1) Wykonać planszę 1920x1080 px z torem w formie zamkniętej pętli jednokierunkowej o szerokości 3-4 m, z co najmniej jednym zakrętem w każdym kierunku. Dopuszczalne są skrzyżowania. Niedopuszczalne jest przygotowanie identycznych planszy przez różne zespoły.
2) Przygotować plik z definicją scenariusza przejazdu dla jednego agenta–wg wskazówek technicznychw dokumencie projekt-wskazowki1.pdf. Planszę i scenariusz dostarczyć wraz z krótkim opisem i uzasadnieniem do 20III. Zainstalować i zapoznać się z dostarczonym środowiskiem symulacyjnym. Liczba punktów –8 (plansza –6, scenariusz –2).

# 2. do 12.V
Wytrenowanie modelu decyzyjnego dla ruchu jednego agenta Wymagania:
1) Zaimplementować algorytm uczenia ze wzmacnianiem dla pojedynczego agenta na podstawie udostępnionego kodu źródłowego.
2) Podjąć próbę poprawy działania poprzez zmianę wartości co najmniej 2 parametrów klas środowiska i klasy uczącej oraz co najmniej 1 parametru lub struktury sieci neuronowej. Wskazówki techniczne w dokumencie projekt-wskazowki2.pdf, wyjściowy kod źródłowy do uzupełnienia i wykorzystania –w plikach *_handout.py. W uczeniu wykorzystywać tylko pomiary udostępniane przez środowisko, w szczególności nie wykorzystywać bezwzględnej lokalizacji agenta. Sieci zapisywać w formacie.tf. Zaimplementować program demonstrujący ruch agenta po trasie. Zarejestrować co najmniej jeden, możliwie najlepszy scenariusz uruchomienia żółwia i przygotować wynik w formie graficznej, z zaznaczonymi krokami na planszy. Przygotować obraz kontenera Docker z wykonaną pracą (symulator, kod źródłowy, modele sieci). Obraz kontenera i krótki opis wykonanych eksperymentów zawierający zestawienie zmian parametrów i uzyskanych wynikówdostarczyć do 8V. Kara za opóźnienie: 1 pkt/dzień.Liczba punktów –17. Wskaźnik oceny jakości modelujest ilorazem liczby okrążeń𝑙i liczby prób 𝑠, tj.: 𝜂=𝑙𝑠., Liczba okrążeń jest szacowana poprzez zliczanie osiągniętych celów pośrednich.Próba jest pojedynczym uruchomieniem agenta ze strategią wg dostarczonego modelu, kierowanego do kolejnych celów pośrednich do czasu wypadnięcia z trasy. Skalaocen:17pkt. dla𝜂>1;14pkt. jeśli 𝜂>0,7;10pkt. jeśli 𝜂>0,5. Nie mniej niż 9pkt. bez względu na 𝜂 jeśli spełniono oba wymagania i dostarczono wszystkie materiały.

# 3. do 12.VI
Uczenie wieloagentowe z uwzględnieniem interakcji agentów Uwaga: dokument od tego miejsca może jeszcze nieznacznie ulec modyfikacjom. Wymagania: 
1) Przygotować planszę wg jednego z wskazanych przez prowadzącego wariantów: a) skrzyżowanie jednokierunkowe –w celu uwzględnienia interakcji na skrzyżowaniu,b.zanikanie i rozdzielanie pasa –w celu uwzględnienia interakcji przy zwężeniu,c.rondo –w celu uwzględnienia pierwszeństwa i oddalania się od celu podczas skrętu w lewo.
2) Przygotować scenariusze przejazdów tak, aby występowała realna szansa wystąpienia kolizji w miejscu manewru na torze (a-c). 
3) Zaimplementować i przeprowadzić uczenie modelu w celu uzyskania możliwie najlepszego wyniku. W tym celu należy dostosować środowisko symulacyjne oraz algorytm DQN do uczenia ustawicznego wielu agentów naraz. Dostosowanie środowiska zasadniczo polega na umożliwieniu selektywnego wznawiania pracy agentów, których epizod treningowy się zakończył, oraz wykonywaniu pojedynczego kroku symulacji dla wszystkich agentów jednocześnie. Dodatkowo należy wychwytywać kolizje, kończąc epizod agenta-sprawcy i naliczając karę. Dostosowanie algorytmu uczenia polega na bieżącym wznawianiu pracy agentów i uczeniu sieci adekwatnie do łącznej liczby kroków wykonanych przez agentyw jednymkroku symulacji.Należy dokonać oceny jakości modelu przy pierwotnej i przy podwojonej liczbie agentów. Następnie należy przeprowadzić ponowne uczenie dla podwojonej liczby agentów i ponownie ocenić model –dla zdwojonej liczby agentów, a także dla pierwotnej ich liczby.Plansze, scenariusze,najlepszy uzyskany model, kod źródłowy i zestawienie wyników należy dostarczyć do 12VI.Liczba punktów –17(planszai scenariusze–4, poprawny kod środowiska i uczenia sieci –4, zestawienie i omówienie wyników –9).

# 4. do 15.VI
Podsumowanie wykonanych prac w formie prezentacji–do 15VI.Liczba punktów –8(czytelność prezentacji, jakość dyskusji o rezultatach; możliwa indywidualna ocena poszczególnych członków zespołów)