## Installation

1. Clone the repository and enter it:

    ```sh
    $ git clone git@github.com:BrainProjectTau/Brain.git #TODO: change the url to the proper one
    ...
    $ cd Brain/
    ```

2. Run the installation script and activate the virtual environment:

    ```sh
    $ ./scripts/install.sh
    ...
    $ source .env/bin/activate
    [Brain] $ # you're good to go!
    ```

3. To check that everything is working as expected, run the tests:

    ```sh
    $ pytest tests/
    ...
    ```

## Usage

The `Brain` packages provides the following classes:

- `Area`
    
    This class represents a general Area which can be binded to use to a specific brain.

    - `Basic Usage` 
        ```python
        >>> from Brain import Area
        >>> amigdala = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
        ```


- `Connectome`
    
    This class represents a simulated connectome which holds the areas, stimuli, and all the synapse weights.
    
    ```python
    >>> from Brain import NonLazyConnectome, LazyConnectome
    >>> amigdala = Area(...)
    >>> hippocampus = Area(...)
    >>> connectome1 = NonLazyconnectome()
    >>> connectome1.add_area(amigdala)
    >>> connectome1.add_area(hippocampus)
    >>> connectome2 = LazyConnectome()
    >>> connectome2.add_area(amigdala)
    >>> connectome2.add_area(hippocampus)
    >>> connectome1.enable(amigdala, hippocampus)
    >>> connectome2.enable(amigdala, hippocampus)
    >>> connectome1.next_round({...}) 
    >>> connectome2.next_round({...}) # Doesn't affect each other
    >>> connectome1.winners(amigdala) # Return connectome1's winners in the amigdala
    ... [...]
    >>> connectome2.winners(amigdala) # Return connectome2's winners in the amigdala
    >>> [...] 
    ```

    - `Full API`
        - `Connectome.add_area(area: Area)`
            Add an Area to the Connectome.

        - `Connectome.add_stimulus(stimulus: Stimulus)`
            Add a Stimulus to the Connectome.

        - `Connectome.enable(source: BrainPart, dest: BrainPart)`
            Enables a connection between 2 BrainParts in the Connectome.

        - `Connectome.enable(source: BrainPart, dest: BrainPart)`
            Disables a connection between 2 BrainParts in the Connectome.

        -  `Connectome.next_round(subconnectome: Connectome = None)`
            Calculate the next set of winners. If subconnectome is mentioned calculates only in it.

        - `Connectome.winners(area: Area)`
            Return the winners in a Area.



- `Assembly`
    
    This class represents an Assembly in the brain.
    

```
To Do:
1. Rename connectome to brain and inform other groups.
(includes changing the names of the classes)
2. Let assemblies and learning describe their APIs and fix them into the document accordingly.

``` ```
