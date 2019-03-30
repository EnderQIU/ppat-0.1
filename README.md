# People & Places Automated Translator

When translating games such as Crusader Kings 2 and Europa Universals 4, there are many names of places and people for
 translators to search in search engines and dictionaries, which will slow down their efficiency and enthusiasm. 
This project is aimed to translate the names automatedly but accurately. CLI mode can read a file to be translated for
 best efficiency and the Web Server can let more translators use it in handle.

## Getting Started

For running this project you need a Python 3 installed and the requisites are listed in `requirements.txt`.
If you have an accelerated graphic card and have the develop environment properly configured,
 you can install requirements from `requirements-gpu.txt`.
You are suggested to have those environments listed below installed for the best experiences:

- **Windows x86-64** with GPU platform: Python3.6(x86-64) tensorflow-gpu==1.12.0 CUDA==9.0 cuDNN==7.3.0.29
- **Windows x86-64** CPU Only platform: Python3.6/7(x86-64) tensorflow==1.12.0/1.13.1
- **Linux x86-64** with GPU platform: Python3.6/7(x86-64) tensorflow-gpu==1.12.0 CUDA==9.0 cuDNN==7.3.0.29
- **Linux x86-64** CPU Only platform: Python3.6/7(x86-64) tensorflow==1.12.0/1.13.1
- **MacOS 10.13 or Later**: Python3.6/7(x86-64) tensorflow==1.12.0/1.13.1

## Usage

### Web Service

```sh
# With requirements already installed
> cd src/
> flask run
```

Check our demo website at <https://ppat.enderqiu.cn>.

### CLI mode

```sh
# With requirements already installed
> cd src/
> python cli.py
... Loading dictionaries and rules
> Alex
Result:
===================================================
From dictionary:
Keyword	Language	Category	Chinese
Alex	葡	People	阿莱士
Alex		People	亚历克斯;亚历克丝(女名)(教名Alexander、Alexandrina的昵称)
Alex	法、美	Places	亚历克斯
---------------------------------------------------
From rule:
Keyword	Language	Category	Chinese
Alex	English	People	阿拉克斯（丝）
Alex	English	Places	阿拉克斯（丝）
===================================================
>
```

Usually results from rules are quite different from those from dictionaries, for the reason
 that some transliterations already exist are transliated by custom or tradition. We should
 consider the former in higer privority than others.

## Supported Languages

- English

You can add and customize transliterating rules following the instructions in
[Transliteration Rules](src/translators/data/README).

## Structure

```ascii
+--------------------------+--------------------------+
|                          |                          |
| Keyword&Parameter Input  |      Result Displayer    |
|                          |                          |
+--------------------------+--------------------------+
|                                                     |
|                    Web Frontend                     |
|                                                     |
+-----------------+----------------^------------------+
                  |                |
+-----------------v----------------+------------------+
|                                                     |
|       Parameter Resolver & Result Formatter         |
|                                                     |
+--------------------------+--------------------------+  Backend
|                          |                          |  Service
|                          |                          |
|                          |                          |
|    Index Transalator     |   Algorithm Translator   |
|                          |                          | <-- Could be a standalone CLI program
|                          |                          |     with another initializer, parameter
|                          |                          |     resolver and result displayer.
+-----^-----------+--------+-------^------------+-----+
      |           |                |            |
+-----+-----------v-------+ +------+------------v-----+
|                         | |                         |
|    Index Data Reader    | |     Algorithm Loader    | Index & Rules
|                         | |                         |   Loader
+-----^-------------------+ +-------------------^-----+
      |                                         |
+-----+------------------------------+ +--------+-----+ Rule & Index
| people.data.json & place.data.json | | t.csv & t.py |  Storage
+------------------------------------+ +--------------+
```

## Translating Process


```
+----------------------+
|                      |
| Raw Input (Words)    |
|                      |
+-----------+----------+  +-------------+
            |             |             |
            +<------------+.to_phonetics|
            |             |             |
+-----------v----------+  +-------------+
|                      |
| Phonetics (Optional) |
|                      |
+-----------+----------+  +-----------+
            |             |           |
            +<------------+ By rules  |
            |             |           |
+-----------v----------+  +-----------+
|                      |
|  Chinese Characters  +--> Post Processes ...
|                      |
+----------------------+
```

According to transliteration tables, some languages have their own pronunciation system, for which we need to translate
 raw words input to phonetics at first and then translate phonetics to Chinese characters. See more process details in 
 [Transliteration Rules](src/translators/data/README). And some rules that can't implement on the table are written
 in `*.py` files.


## Authors

- [EnderQIU](https://github.com/EnderQIU)

## License

This project is open-sourced under the GPL v3 License.

## About Copyright

1. No copyright infringement intended.
2. If there is a copyright violation content, please e-mail <a934560824@163.com> to delete.
3. No money is being made of this project.
