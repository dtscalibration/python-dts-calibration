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


.. parsed-literal::

    /Volumes/GoogleDrive/My Drive/Projects/python-dts-calibration/src/dtscalibration/io.py:1835: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_localize(


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
    Dimensions:                (x: 11582, time: 3, trans_att: 0)
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
      display: block !important;
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
    Dimensions:                (x: 11582, time: 3, trans_att: 0)
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
    
    .. and many more attributes. See: ds.attrs</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataStore</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-2fb81a69-b0da-45a9-9fd9-c694beb14370' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-2fb81a69-b0da-45a9-9fd9-c694beb14370' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 11582</li><li><span class='xr-has-index'>time</span>: 3</li><li><span class='xr-has-index'>trans_att</span>: 0</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-ccbd34da-988f-4c79-af20-bc1844370007' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ccbd34da-988f-4c79-af20-bc1844370007' class='xr-section-summary' >Coordinates: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-451.4 -450.9 ... 5.409e+03</div><input id='attrs-3a095f3c-70da-4526-8c97-dbcdd0646d28' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-3a095f3c-70da-4526-8c97-dbcdd0646d28' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-457d2ba2-afdb-4dd7-ba39-f4fa96d2e05b' class='xr-var-data-in' type='checkbox'><label for='data-457d2ba2-afdb-4dd7-ba39-f4fa96d2e05b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>distance</dd><dt><span>description :</span></dt><dd>Length along fiber</dd><dt><span>long_description :</span></dt><dd>Starting at connector of forward channel</dd><dt><span>units :</span></dt><dd>m</dd></dl></div><div class='xr-var-data'><pre>array([-451.37958, -450.87354, -450.3675 , ..., 5407.952  , 5408.4585 ,
           5408.9644 ], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>filename</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U25</div><div class='xr-var-preview xr-preview'>&#x27;15_56_47_BinaryRawDTS.dat&#x27; ... ...</div><input id='attrs-743e104b-b030-48e6-ae06-93cdb9ca0755' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-743e104b-b030-48e6-ae06-93cdb9ca0755' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dbee34c9-2ad2-4c1d-a0f1-afe894f14551' class='xr-var-data-in' type='checkbox'><label for='data-dbee34c9-2ad2-4c1d-a0f1-afe894f14551' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;15_56_47_BinaryRawDTS.dat&#x27;, &#x27;16_11_31_BinaryRawDTS.dat&#x27;,
           &#x27;16_29_23_BinaryRawDTS.dat&#x27;], dtype=&#x27;&lt;U25&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>filename_temp</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;15_56_47_BinaryTemp.dat&#x27; ... &#x27;1...</div><input id='attrs-c3b9c2a7-d500-44df-9362-7cf14ec1404d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c3b9c2a7-d500-44df-9362-7cf14ec1404d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4dc0e84c-0f9b-4b80-ad4e-9a09577cfb1a' class='xr-var-data-in' type='checkbox'><label for='data-4dc0e84c-0f9b-4b80-ad4e-9a09577cfb1a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;15_56_47_BinaryTemp.dat&#x27;, &#x27;16_11_31_BinaryTemp.dat&#x27;,
           &#x27;16_29_23_BinaryTemp.dat&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>timestart</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2009-09-24T00:56:46 ... 2009-09-...</div><input id='attrs-cb64411c-bb01-4430-adb7-2035bb89c8fe' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-cb64411c-bb01-4430-adb7-2035bb89c8fe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9a3a59ea-ee4d-43ad-8da4-8319c525371f' class='xr-var-data-in' type='checkbox'><label for='data-9a3a59ea-ee4d-43ad-8da4-8319c525371f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time start of the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2009-09-24T00:56:46.000000000&#x27;, &#x27;2009-09-24T01:11:30.000000000&#x27;,
           &#x27;2009-09-24T01:29:22.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>timeend</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2009-09-24T00:56:47 ... 2009-09-...</div><input id='attrs-463d7fc2-57ae-4bc5-9f6e-92eac6e88f34' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-463d7fc2-57ae-4bc5-9f6e-92eac6e88f34' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-13028783-9ca3-42cf-87b4-d712beb120bd' class='xr-var-data-in' type='checkbox'><label for='data-13028783-9ca3-42cf-87b4-d712beb120bd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time end of the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2009-09-24T00:56:47.000000000&#x27;, &#x27;2009-09-24T01:11:31.000000000&#x27;,
           &#x27;2009-09-24T01:29:23.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2009-09-24T00:56:47 ... 2009-09-...</div><input id='attrs-aed40a28-1b21-4769-8787-5f9097004f49' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-aed40a28-1b21-4769-8787-5f9097004f49' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a0ae9dea-03f5-488a-8cbc-ed94e2e9bb28' class='xr-var-data-in' type='checkbox'><label for='data-a0ae9dea-03f5-488a-8cbc-ed94e2e9bb28' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time halfway the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2009-09-24T00:56:47.000000000&#x27;, &#x27;2009-09-24T01:11:31.000000000&#x27;,
           &#x27;2009-09-24T01:29:23.000000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>acquisitiontimeFW</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>timedelta64[ns]</div><div class='xr-var-preview xr-preview'>00:00:01 00:00:01 00:00:01</div><input id='attrs-9e846bf0-8fc5-400a-893e-5347e8a6da37' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-9e846bf0-8fc5-400a-893e-5347e8a6da37' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d74a5a98-287c-4222-81d3-eda16d3dd58d' class='xr-var-data-in' type='checkbox'><label for='data-d74a5a98-287c-4222-81d3-eda16d3dd58d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Acquisition time of the forward measurement</dd></dl></div><div class='xr-var-data'><pre>array([1000000000, 1000000000, 1000000000], dtype=&#x27;timedelta64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>trans_att</span></div><div class='xr-var-dims'>(trans_att)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'></div><input id='attrs-8cd1e910-e93c-44f1-8d5c-3c2c665bdf78' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-8cd1e910-e93c-44f1-8d5c-3c2c665bdf78' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d7b14049-17c5-4fdc-848b-5696405ff195' class='xr-var-data-in' type='checkbox'><label for='data-d7b14049-17c5-4fdc-848b-5696405ff195' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>Locations introducing transient directional differential attenuation</dd><dt><span>description :</span></dt><dd>Locations along the x-dimension introducing transient directional differential attenuation</dd><dt><span>long_description :</span></dt><dd>Connectors introduce additional differential attenuation that is different for the forward and backward direction, and varies over time.</dd><dt><span>units :</span></dt><dd>m</dd></dl></div><div class='xr-var-data'><pre>array([], dtype=float64)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-67d4a72d-e23f-4f78-a8d8-a1e1d9ca0968' class='xr-section-summary-in' type='checkbox'  checked><label for='section-67d4a72d-e23f-4f78-a8d8-a1e1d9ca0968' class='xr-section-summary' >Data variables: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>st</span></div><div class='xr-var-dims'>(x, time)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>39040680 39057147 ... 39071213</div><input id='attrs-c78b2d29-a94c-499b-aa06-d962dcc7b786' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-c78b2d29-a94c-499b-aa06-d962dcc7b786' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bcecb575-8caf-41d3-8aa3-a5a3b4bf0523' class='xr-var-data-in' type='checkbox'><label for='data-bcecb575-8caf-41d3-8aa3-a5a3b4bf0523' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>st</dd><dt><span>description :</span></dt><dd>Stokes intensity</dd><dt><span>units :</span></dt><dd>-</dd></dl></div><div class='xr-var-data'><pre>array([[39040680, 39057147, 39067220],
           [39038580, 39053177, 39063543],
           [39038768, 39054349, 39064780],
           ...,
           [39155768, 39179638, 39196217],
           [39046316, 39063478, 39073966],
           [39046948, 39061160, 39071213]], dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ast</span></div><div class='xr-var-dims'>(x, time)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>39048646 39064414 ... 39407668</div><input id='attrs-c39a848d-5035-46c9-abec-9f84e930ef2b' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-c39a848d-5035-46c9-abec-9f84e930ef2b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-58c2b076-c7e4-47cd-924e-5110425ccd4f' class='xr-var-data-in' type='checkbox'><label for='data-58c2b076-c7e4-47cd-924e-5110425ccd4f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>ast</dd><dt><span>description :</span></dt><dd>anti-Stokes intensity</dd><dt><span>units :</span></dt><dd>-</dd></dl></div><div class='xr-var-data'><pre>array([[39048646, 39064414, 39074033],
           [39046719, 39060574, 39071003],
           [39046655, 39061723, 39072593],
           ...,
           [39304136, 39313172, 39321329],
           [39461032, 39474405, 39483689],
           [39362443, 39388893, 39407668]], dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>tmp</span></div><div class='xr-var-dims'>(x, time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-273.1 -273.1 ... 82.41 82.71</div><input id='attrs-e968df11-d188-4a4a-917e-180e075bb2f5' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-e968df11-d188-4a4a-917e-180e075bb2f5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-575b47c0-95c1-43ec-accc-dce7c8e64730' class='xr-var-data-in' type='checkbox'><label for='data-575b47c0-95c1-43ec-accc-dce7c8e64730' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>tmp</dd><dt><span>description :</span></dt><dd>Temperature calibrated by device</dd><dt><span>units :</span></dt><dd>degC</dd></dl></div><div class='xr-var-data'><pre>array([[-273.1499939 , -273.1499939 , -273.1499939 ],
           [-273.1499939 , -273.1499939 , -273.1499939 ],
           [-273.1499939 , -273.1499939 , -273.1499939 ],
           ...,
           [  51.81999969,   49.75999832,   48.02000046],
           [  74.16000366,   73.44999695,   72.34999847],
           [  80.91000366,   82.41000366,   82.70999908]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>referenceTemperature</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>28.61 29.24 30.29</div><input id='attrs-3bdfff36-e0db-488f-99e5-b4d0c8c2cfcc' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-3bdfff36-e0db-488f-99e5-b4d0c8c2cfcc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ea2de04d-b132-48af-8259-64ae68bc1833' class='xr-var-data-in' type='checkbox'><label for='data-ea2de04d-b132-48af-8259-64ae68bc1833' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>reference temperature</dd><dt><span>description :</span></dt><dd>Internal reference temperature</dd><dt><span>units :</span></dt><dd>degC</dd></dl></div><div class='xr-var-data'><pre>array([28.61147461, 29.23735962, 30.29247437])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>st_zero</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>3.904e+07 3.906e+07 3.907e+07</div><input id='attrs-6d2d6c26-2512-4f30-83ab-0c9b9c451786' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-6d2d6c26-2512-4f30-83ab-0c9b9c451786' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-12974864-4460-42e4-a5aa-9fd65be6a57e' class='xr-var-data-in' type='checkbox'><label for='data-12974864-4460-42e4-a5aa-9fd65be6a57e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>ST_zero</dd><dt><span>description :</span></dt><dd>Stokes zero count</dd><dt><span>units :</span></dt><dd>counts</dd></dl></div><div class='xr-var-data'><pre>array([39042026.18660287, 39057430.34449761, 39067731.48325359])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ast_zero</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>3.905e+07 3.907e+07 3.908e+07</div><input id='attrs-ab3a8e30-245c-47e2-9b83-13c9e337f882' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-ab3a8e30-245c-47e2-9b83-13c9e337f882' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-27310fe0-6349-4134-91c6-dfb736f133c4' class='xr-var-data-in' type='checkbox'><label for='data-27310fe0-6349-4134-91c6-dfb736f133c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>AST_zero</dd><dt><span>description :</span></dt><dd>anit-Stokes zero count</dd><dt><span>units :</span></dt><dd>counts</dd></dl></div><div class='xr-var-data'><pre>array([39050438.62200957, 39065503.        , 39075698.97607656])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>userAcquisitionTimeFW</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.0 1.0 1.0</div><input id='attrs-6124bf97-de75-489e-8ac0-e2ae52974390' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-6124bf97-de75-489e-8ac0-e2ae52974390' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ce1e8b51-ebeb-42c5-9392-3b27a4823a1c' class='xr-var-data-in' type='checkbox'><label for='data-ce1e8b51-ebeb-42c5-9392-3b27a4823a1c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>userAcquisitionTimeFW</dd><dt><span>description :</span></dt><dd>Measurement duration of forward channel</dd><dt><span>long_description :</span></dt><dd>Desired measurement duration of forward channel</dd><dt><span>units :</span></dt><dd>seconds</dd></dl></div><div class='xr-var-data'><pre>array([1., 1., 1.])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-79710f24-6684-480b-be77-ba58bee83ed3' class='xr-section-summary-in' type='checkbox'  ><label for='section-79710f24-6684-480b-be77-ba58bee83ed3' class='xr-section-summary' >Attributes: <span>(16)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>survey_type :</span></dt><dd>2</dd><dt><span>hdr_version :</span></dt><dd>3</dd><dt><span>x_units :</span></dt><dd>n/a</dd><dt><span>y_units :</span></dt><dd>counts</dd><dt><span>num_points :</span></dt><dd>12000</dd><dt><span>num_pulses :</span></dt><dd>25000</dd><dt><span>channel_id :</span></dt><dd>1</dd><dt><span>num_subtraces :</span></dt><dd>354</dd><dt><span>num_skipped :</span></dt><dd>0</dd><dt><span>probe_name :</span></dt><dd>walla1</dd><dt><span>hdr_size :</span></dt><dd>176</dd><dt><span>hw_config :</span></dt><dd>84</dd><dt><span>isDoubleEnded :</span></dt><dd>0</dd><dt><span>forwardMeasurementChannel :</span></dt><dd>0</dd><dt><span>backwardMeasurementChannel :</span></dt><dd>N/A</dd><dt><span>_sections :</span></dt><dd>null
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

    <matplotlib.lines.Line2D at 0x1425ae970>



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



