import csv

'''Kod dziala dla jednego scenariusza w w pliu .csv jesli okaze sie ze trzeba 
bedzie dodac kilka scenariuszy do pliku .csv to trzeba bedzie zmodyfikowac kod aby je parsowal poprawnie.
print'y sa do sprawdzenia jak dziala to co napisalem, jesli ktos jest ciekaw'''

routes={} # trasy agentów {id_trasy:(liczba_agentów,xmin,xmax,ymin,ymax,xg,yg)}

routes_fname="scenario.csv"

with open(routes_fname,encoding='utf-8-sig') as f:  # załadowanie tras agentów
    test = csv.reader(f)
    sequences=[]

    for row in test:
        # print(row)
        split_row=row[0].split(';')
        # print(split_row)
        split_row_ints = list(map(int, split_row))      # Map items in list from 'str' to 'int' 
        route_id=split_row_ints[0]
        sequences.append(split_row_ints[1:])

    routes[route_id]=sequences
    # print(routes)
    f.close()

for route,sections in routes.items():               # dla kolejnych tras
    # print(f'Route: {route} Sections: {sections}')
    for sec_id,sec in enumerate(sections):          # dla kolejnych odcinków trasy
        # print(f'id: {sec_id} section: {sec}')
        for seq in range(sec[0]):                   # utwórz określoną liczbę żółwi
            tname=f'{route}_{sec_id}_{seq}'         # identyfikator agenta: trasa, segment pocz., nr kolejny
            # print(f'Agent {tname}')