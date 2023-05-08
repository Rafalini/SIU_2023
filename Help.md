# Montowanie


```docker run --name siu -p 6080:80 -e RESOLUTION=1366x768 --mount type=bind,source="$(pwd)",target=/root/konrad dudekw/siu-20.04 ```

Konrad
flaga --mount, type bind, w source podajesz folder hosta w target podajesz na co ma się ten folder mapować w kontenerze, można podać też parametr readonly żeby z kontenera nie dało się tam nic zmieniać
Konrad

ref: https://docs.docker.com/storage/bind-mounts/
