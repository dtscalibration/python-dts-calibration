A4. Loading sensortran files
============================

This example loads sensortran files. Only single-ended measurements are
currently supported. Sensortran files are in binary format. The library
requires the ``*BinaryRawDTS.dat`` and ``*BinaryTemp.dat`` files.

.. code:: ipython3

    import os
    import glob
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
        
    from dtscalibration import read_sensortran_files

The example data files are located in
``./python-dts-calibration/tests/data``.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'sensortran_binary')
    print(filepath)


.. parsed-literal::

    ../../tests/data/sensortran_binary


.. code:: ipython3

    filepathlist = sorted(glob.glob(os.path.join(filepath, '*.dat')))
    filenamelist = [os.path.basename(path) for path in filepathlist]
    
    for fn in filenamelist:
        print(fn)


.. parsed-literal::

    15_56_47_BinaryRawDTS.dat
    15_56_47_BinaryTemp.dat
    16_11_31_BinaryRawDTS.dat
    16_11_31_BinaryTemp.dat
    16_29_23_BinaryRawDTS.dat
    16_29_23_BinaryTemp.dat


We will simply load in the binary files

.. code:: ipython3

    ds = read_sensortran_files(directory=filepath)


.. parsed-literal::

    3 files were found, each representing a single timestep
    Recorded at 11582 points along the cable
    The measurement is single ended


The object tries to gather as much metadata from the measurement files
as possible (temporal and spatial coordinates, filenames, temperature
probes measurements). All other configuration settings are loaded from
the first files and stored as attributes of the ``DataStore``.
Sensortran’s data files contain less information than the other
manufacturer’s devices, one being the acquisition time. The acquisition
time is needed for estimating variances, and is set a constant 1s.

.. code:: ipython3

    print(ds)


.. parsed-literal::

    <dtscalibration.DataStore>
    Sections:                  ()
    Dimensions:                (time: 3, trans_att: 0, x: 11582)
    Coordinates:
      * x                      (x) float32 -451.4 -450.9 ... 5.408e+03 5.409e+03
        filename               (time) <U25 '15_56_47_BinaryRawDTS.dat' ... '16_29...
        filename_temp          (time) <U23 '15_56_47_BinaryTemp.dat' ... '16_29_2...
        timestart              (time) datetime64[ns] 2009-09-24T00:56:46 ... 2009...
        timeend                (time) datetime64[ns] 2009-09-24T00:56:47 ... 2009...
      * time                   (time) datetime64[ns] 2009-09-24T00:56:47 ... 2009...
        acquisitiontimeFW      (time) timedelta64[ns] 00:00:01 00:00:01 00:00:01
      * trans_att              (trans_att) float64 
    Data variables:
        st                     (x, time) int32 39040680 39057147 ... 39071213
        ast                    (x, time) int32 39048646 39064414 ... 39407668
        tmp                    (x, time) float64 -273.1 -273.1 ... 82.41 82.71
        referenceTemperature   (time) float64 28.61 29.24 30.29
        st_zero                (time) float64 3.904e+07 3.906e+07 3.907e+07
        ast_zero               (time) float64 3.905e+07 3.907e+07 3.908e+07
        userAcquisitionTimeFW  (time) float64 1.0 1.0 1.0
    Attributes: (12/16)
        survey_type:                 2
        hdr_version:                 3
        x_units:                     n/a
        y_units:                     counts
        num_points:                  12000
        num_pulses:                  25000
        ...                          ...
        hdr_size:                    176
        hw_config:                   84
    
    .. and many more attributes. See: ds.attrs


The sensortran files differ from other manufacturers, in that they
return the ‘counts’ of the Stokes and anti-Stokes signals. These are not
corrected for offsets, which has to be done manually for proper
calibration.

Based on the data available in the binary files, the library estimates a
zero-count to correct the signals, but this is not perfectly accurate or
constant over time. For proper calibration, the offsets would have to be
incorporated into the calibration routine.

.. code:: ipython3

    ds




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */
    
    :root {
      --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
      --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
      --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
      --xr-border-color: var(--jp-border-color2, #e0e0e0);
      --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
      --xr-background-color: var(--jp-layout-color0, white);
      --xr-background-color-row-even: var(--jp-layout-color1, white);
      --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
    }
    
    html[theme=dark],
    body.vscode-dark {
      --xr-font-color0: rgba(255, 255, 255, 1);
      --xr-font-color2: rgba(255, 255, 255, 0.54);
      --xr-font-color3: rgba(255, 255, 255, 0.38);
      --xr-border-color: #1F1F1F;
      --xr-disabled-color: #515151;
      --xr-background-color: #111111;
      --xr-background-color-row-even: #111111;
      --xr-background-color-row-odd: #313131;
    }
    
    .xr-wrap {
      display: block;
      min-width: 300px;
      max-width: 700px;
    }
    
    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type {
      color: var(--xr-font-color2);
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 20px 20px;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item input {
      display: none;
    }
    
    .xr-section-item input + label {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: '►';
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: '▼';
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: '(';
    }
    
    .xr-dim-list:after {
      content: ')';
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ',';
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      margin-bottom: 0;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data {
      display: none;
      background-color: var(--xr-background-color) !important;
      padding-bottom: 5px !important;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2 {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;dtscalibration.DataStore&gt;
    Sections:                  ()
    Dimensions:                (time: 3, trans_att: 0, x: 11582)
    Coordinates:
      * x                      (x) float32 -451.4 -450.9 ... 5.408e+03 5.409e+03
        filename               (time) &lt;U25 &#x27;15_56_47_BinaryRawDTS.dat&#x27; ... &#x27;16_29...
        filename_temp          (time) &lt;U23 &#x27;15_56_47_BinaryTemp.dat&#x27; ... &#x27;16_29_2...
        timestart              (time) datetime64[ns] 2009-09-24T00:56:46 ... 2009...
        timeend                (time) datetime64[ns] 2009-09-24T00:56:47 ... 2009...
      * time                   (time) datetime64[ns] 2009-09-24T00:56:47 ... 2009...
        acquisitiontimeFW      (time) timedelta64[ns] 00:00:01 00:00:01 00:00:01
      * trans_att              (trans_att) float64 
    Data variables:
        st                     (x, time) int32 39040680 39057147 ... 39071213
        ast                    (x, time) int32 39048646 39064414 ... 39407668
        tmp                    (x, time) float64 -273.1 -273.1 ... 82.41 82.71
        referenceTemperature   (time) float64 28.61 29.24 30.29
        st_zero                (time) float64 3.904e+07 3.906e+07 3.907e+07
        ast_zero               (time) float64 3.905e+07 3.907e+07 3.908e+07
        userAcquisitionTimeFW  (time) float64 1.0 1.0 1.0
    Attributes: (12/16)
        survey_type:                 2
        hdr_version:                 3
        x_units:                     n/a
        y_units:                     counts
        num_points:                  12000
        num_pulses:                  25000
        ...                          ...
        hdr_size:                    176
        hw_config:                   84
    
    .. and many more attributes. See: ds.attrs</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.DataStore</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-5d8ea54a-a6ec-4884-8235-7d8bbff6659e' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-5d8ea54a-a6ec-4884-8235-7d8bbff6659e' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 3</li><li><span class='xr-has-index'>trans_att</span>: 0</li><li><span class='xr-has-index'>x</span>: 11582</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-2ba13c5f-c082-467f-bea2-7de3c7ecb127' class='xr-section-summary-in' type='checkbox'  checked><label for='section-2ba13c5f-c082-467f-bea2-7de3c7ecb127' class='xr-section-summary' >Coordinates: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-451.4 -450.9 ... 5.409e+03</div><input id='attrs-07594804-1e61-4533-8069-d174c658036b' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-07594804-1e61-4533-8069-d174c658036b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4870d243-003a-4f48-85f1-8cd97f7d7b99' class='xr-var-data-in' type='checkbox'><label for='data-4870d243-003a-4f48-85f1-8cd97f7d7b99' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>distance</dd><dt><span>description :</span></dt><dd>Length along fiber</dd><dt><span>long_description :</span></dt><dd>Starting at connector of forward channel</dd><dt><span>units :</span></dt><dd>m</dd></dl></div><div class='xr-var-data'><pre>array([-451.37958, -450.87354, -450.3675 , ..., 5407.952  , 5408.4585 ,
           5408.9644 ], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>filename</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U25</div><div class='xr-var-preview xr-preview'>&#x27;15_56_47_BinaryRawDTS.dat&#x27; ... ...</div><input id='attrs-af1a3f2a-0ce9-4b36-8ba8-830be2ac93d1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-af1a3f2a-0ce9-4b36-8ba8-830be2ac93d1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eaa77a63-23d1-4786-a7fd-4f074f301970' class='xr-var-data-in' type='checkbox'><label for='data-eaa77a63-23d1-4786-a7fd-4f074f301970' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;15_56_47_BinaryRawDTS.dat&#x27;, &#x27;16_11_31_BinaryRawDTS.dat&#x27;,
           &#x27;16_29_23_BinaryRawDTS.dat&#x27;], dtype=&#x27;&lt;U25&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>filename_temp</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;15_56_47_BinaryTemp.dat&#x27; ... &#x27;1...</div><input id='attrs-01531941-a903-41cd-a88a-ccbe31b8050e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-01531941-a903-41cd-a88a-ccbe31b8050e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e61b2243-b248-4ddd-a432-4ffe954908c0' class='xr-var-data-in' type='checkbox'><label for='data-e61b2243-b248-4ddd-a432-4ffe954908c0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;15_56_47_BinaryTemp.dat&#x27;, &#x27;16_11_31_BinaryTemp.dat&#x27;,
           &#x27;16_29_23_BinaryTemp.dat&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>timestart</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2009-09-24T00:56:46 ... 2009-09-...</div><input id='attrs-5e956ffa-d46d-4906-948b-7b8bc7857fce' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-5e956ffa-d46d-4906-948b-7b8bc7857fce' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f468d6f6-0458-4d41-868a-d83f5b5b52de' class='xr-var-data-in' type='checkbox'><label for='data-f468d6f6-0458-4d41-868a-d83f5b5b52de' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time start of the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2009-09-24T00:56:46.000000000&#x27;, &#x27;2009-09-24T01:11:30.000000000&#x27;,
           &#x27;2009-09-24T01:29:22.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>timeend</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2009-09-24T00:56:47 ... 2009-09-...</div><input id='attrs-37f049bb-ba56-4403-9f69-beb1c897336c' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-37f049bb-ba56-4403-9f69-beb1c897336c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c4873e31-6055-47e9-815e-c5da5921e4e4' class='xr-var-data-in' type='checkbox'><label for='data-c4873e31-6055-47e9-815e-c5da5921e4e4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time end of the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2009-09-24T00:56:47.000000000&#x27;, &#x27;2009-09-24T01:11:31.000000000&#x27;,
           &#x27;2009-09-24T01:29:23.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2009-09-24T00:56:47 ... 2009-09-...</div><input id='attrs-83c66f0d-e937-4fcb-8a6b-058df60c3569' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-83c66f0d-e937-4fcb-8a6b-058df60c3569' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7a57ce1c-ca03-4cc1-9247-10ba7ec57b9b' class='xr-var-data-in' type='checkbox'><label for='data-7a57ce1c-ca03-4cc1-9247-10ba7ec57b9b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time halfway the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2009-09-24T00:56:47.000000000&#x27;, &#x27;2009-09-24T01:11:31.000000000&#x27;,
           &#x27;2009-09-24T01:29:23.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>acquisitiontimeFW</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>timedelta64[ns]</div><div class='xr-var-preview xr-preview'>00:00:01 00:00:01 00:00:01</div><input id='attrs-1c2abd90-2c78-488d-8518-78b8e4c8a14c' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-1c2abd90-2c78-488d-8518-78b8e4c8a14c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cc074299-6dec-43d8-8c24-9ea5cffa6157' class='xr-var-data-in' type='checkbox'><label for='data-cc074299-6dec-43d8-8c24-9ea5cffa6157' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Acquisition time of the forward measurement</dd></dl></div><div class='xr-var-data'><pre>array([1000000000, 1000000000, 1000000000], dtype=&#x27;timedelta64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>trans_att</span></div><div class='xr-var-dims'>(trans_att)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'></div><input id='attrs-069521b8-33a8-4edd-adc9-28954c70b0be' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-069521b8-33a8-4edd-adc9-28954c70b0be' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0780c1fc-25b0-4465-a78a-96bee571b284' class='xr-var-data-in' type='checkbox'><label for='data-0780c1fc-25b0-4465-a78a-96bee571b284' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>Locations introducing transient directional differential attenuation</dd><dt><span>description :</span></dt><dd>Locations along the x-dimension introducing transient directional differential attenuation</dd><dt><span>long_description :</span></dt><dd>Connectors introduce additional differential attenuation that is different for the forward and backward direction, and varies over time.</dd><dt><span>units :</span></dt><dd>m</dd></dl></div><div class='xr-var-data'><pre>array([], dtype=float64)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f8a564a3-b635-4684-9d83-1bb6334228af' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f8a564a3-b635-4684-9d83-1bb6334228af' class='xr-section-summary' >Data variables: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>st</span></div><div class='xr-var-dims'>(x, time)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>39040680 39057147 ... 39071213</div><input id='attrs-d8ba4c42-a4f9-4131-8fac-f7679b53cb75' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-d8ba4c42-a4f9-4131-8fac-f7679b53cb75' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9604c525-d027-4e8b-93e9-06daec3b7195' class='xr-var-data-in' type='checkbox'><label for='data-9604c525-d027-4e8b-93e9-06daec3b7195' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>st</dd><dt><span>description :</span></dt><dd>Stokes intensity</dd><dt><span>units :</span></dt><dd>-</dd></dl></div><div class='xr-var-data'><pre>array([[39040680, 39057147, 39067220],
           [39038580, 39053177, 39063543],
           [39038768, 39054349, 39064780],
           ...,
           [39155768, 39179638, 39196217],
           [39046316, 39063478, 39073966],
           [39046948, 39061160, 39071213]], dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ast</span></div><div class='xr-var-dims'>(x, time)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>39048646 39064414 ... 39407668</div><input id='attrs-65fbf64f-eea8-46f1-828c-329dcb0f68c9' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-65fbf64f-eea8-46f1-828c-329dcb0f68c9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d45dc2a0-5f7b-4858-b950-3d1b08b9e5df' class='xr-var-data-in' type='checkbox'><label for='data-d45dc2a0-5f7b-4858-b950-3d1b08b9e5df' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>ast</dd><dt><span>description :</span></dt><dd>anti-Stokes intensity</dd><dt><span>units :</span></dt><dd>-</dd></dl></div><div class='xr-var-data'><pre>array([[39048646, 39064414, 39074033],
           [39046719, 39060574, 39071003],
           [39046655, 39061723, 39072593],
           ...,
           [39304136, 39313172, 39321329],
           [39461032, 39474405, 39483689],
           [39362443, 39388893, 39407668]], dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>tmp</span></div><div class='xr-var-dims'>(x, time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-273.1 -273.1 ... 82.41 82.71</div><input id='attrs-ffcd69e8-6716-4914-a3fc-ab1c4aef0d54' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-ffcd69e8-6716-4914-a3fc-ab1c4aef0d54' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d4c28f99-9d88-41e1-bfe4-3520ba06f355' class='xr-var-data-in' type='checkbox'><label for='data-d4c28f99-9d88-41e1-bfe4-3520ba06f355' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>tmp</dd><dt><span>description :</span></dt><dd>Temperature calibrated by device</dd><dt><span>units :</span></dt><dd>degC</dd></dl></div><div class='xr-var-data'><pre>array([[-273.1499939 , -273.1499939 , -273.1499939 ],
           [-273.1499939 , -273.1499939 , -273.1499939 ],
           [-273.1499939 , -273.1499939 , -273.1499939 ],
           ...,
           [  51.81999969,   49.75999832,   48.02000046],
           [  74.16000366,   73.44999695,   72.34999847],
           [  80.91000366,   82.41000366,   82.70999908]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>referenceTemperature</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>28.61 29.24 30.29</div><input id='attrs-517aa817-b730-4d7f-82cb-64ea976777c1' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-517aa817-b730-4d7f-82cb-64ea976777c1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ad57ff10-95d9-4f08-8939-b8687ffe62d2' class='xr-var-data-in' type='checkbox'><label for='data-ad57ff10-95d9-4f08-8939-b8687ffe62d2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>reference temperature</dd><dt><span>description :</span></dt><dd>Internal reference temperature</dd><dt><span>units :</span></dt><dd>degC</dd></dl></div><div class='xr-var-data'><pre>array([28.61147461, 29.23735962, 30.29247437])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>st_zero</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>3.904e+07 3.906e+07 3.907e+07</div><input id='attrs-4bcf9e23-0684-4963-a733-a13cfd4e2443' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-4bcf9e23-0684-4963-a733-a13cfd4e2443' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d70c1201-a880-4b31-9a57-8f57195d5602' class='xr-var-data-in' type='checkbox'><label for='data-d70c1201-a880-4b31-9a57-8f57195d5602' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>ST_zero</dd><dt><span>description :</span></dt><dd>Stokes zero count</dd><dt><span>units :</span></dt><dd>counts</dd></dl></div><div class='xr-var-data'><pre>array([39042026.18660287, 39057430.34449761, 39067731.48325359])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ast_zero</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>3.905e+07 3.907e+07 3.908e+07</div><input id='attrs-65219819-28f4-4f21-8545-23e9f203e62e' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-65219819-28f4-4f21-8545-23e9f203e62e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4065ffcc-f6cc-4050-8dc7-6a5e49fcf04f' class='xr-var-data-in' type='checkbox'><label for='data-4065ffcc-f6cc-4050-8dc7-6a5e49fcf04f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>AST_zero</dd><dt><span>description :</span></dt><dd>anit-Stokes zero count</dd><dt><span>units :</span></dt><dd>counts</dd></dl></div><div class='xr-var-data'><pre>array([39050438.62200957, 39065503.        , 39075698.97607656])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>userAcquisitionTimeFW</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.0 1.0 1.0</div><input id='attrs-e2cd2698-82e5-4cfd-b526-c6c391818fcd' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-e2cd2698-82e5-4cfd-b526-c6c391818fcd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-69950667-0ea7-469d-94ad-7eec96151dcc' class='xr-var-data-in' type='checkbox'><label for='data-69950667-0ea7-469d-94ad-7eec96151dcc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>userAcquisitionTimeFW</dd><dt><span>description :</span></dt><dd>Measurement duration of forward channel</dd><dt><span>long_description :</span></dt><dd>Desired measurement duration of forward channel</dd><dt><span>units :</span></dt><dd>seconds</dd></dl></div><div class='xr-var-data'><pre>array([1., 1., 1.])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-db5a017c-ba40-4253-b241-3bd245f23420' class='xr-section-summary-in' type='checkbox'  ><label for='section-db5a017c-ba40-4253-b241-3bd245f23420' class='xr-section-summary' >Attributes: <span>(16)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>survey_type :</span></dt><dd>2</dd><dt><span>hdr_version :</span></dt><dd>3</dd><dt><span>x_units :</span></dt><dd>n/a</dd><dt><span>y_units :</span></dt><dd>counts</dd><dt><span>num_points :</span></dt><dd>12000</dd><dt><span>num_pulses :</span></dt><dd>25000</dd><dt><span>channel_id :</span></dt><dd>1</dd><dt><span>num_subtraces :</span></dt><dd>354</dd><dt><span>num_skipped :</span></dt><dd>0</dd><dt><span>probe_name :</span></dt><dd>walla1</dd><dt><span>hdr_size :</span></dt><dd>176</dd><dt><span>hw_config :</span></dt><dd>84</dd><dt><span>isDoubleEnded :</span></dt><dd>0</dd><dt><span>forwardMeasurementChannel :</span></dt><dd>0</dd><dt><span>backwardMeasurementChannel :</span></dt><dd>N/A</dd><dt><span>_sections :</span></dt><dd>null
    ...
    </dd></dl></div></li></ul></div></div>



.. code:: ipython3

    ds0 = ds.isel(time=0)
    
    plt.figure()
    ds0.st.plot(label='Stokes signal')
    plt.axhline(ds0.st_zero.values, c='r', label="'zero' measurement")
    plt.legend()
    plt.title('')
    plt.axhline(c='k')




.. parsed-literal::

    <matplotlib.lines.Line2D at 0x7f9af52b5c10>



After a correction and rescaling (for human readability) the data will
look more like other manufacturer’s devices

.. code:: ipython3

    ds['st'] = (ds.st - ds.st_zero)/1e4
    ds['ast'] = (ds.ast - ds.ast_zero)/1e4

.. code:: ipython3

    ds.isel(time=0).st.plot(label='Stokes intensity')
    ds.isel(time=0).ast.plot(label='anti-Stokes intensity')
    plt.legend()
    plt.axhline(c='k', lw=1)
    plt.xlabel('')
    plt.title('')
    plt.ylim([-50,500])




.. parsed-literal::

    (-50.0, 500.0)



