2. Common DataStore functions
=============================

Examples of how to do some of the more commonly used functions:

1. mean, min, max, std
2. Selecting
3. Selecting by index
4. Downsample (time dimension)
5. Upsample / Interpolation (length and time dimension)

.. code:: ipython3

    import os
    
    from dtscalibration import read_silixa_files

First we load the raw measurements into a ``DataStore`` object, as we
learned from the previous notebook.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'single_ended')
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')


.. parsed-literal::

    3 files were found, each representing a single timestep
    4 recorded vars were found: LAF, ST, AST, TMP
    Recorded at 1461 points along the cable
    The measurement is single ended
    Reading the data from disk


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/io.py:1843: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_convert(timezone_netcdf).astype(


0 Access the data
-----------------

The implemented read routines try to read as much data from the raw DTS
files as possible. Usually they would have coordinates (time and space)
and Stokes and anti Stokes measurements. We can access the data by key.
It is presented as a DataArray. More examples are found at
http://xarray.pydata.org/en/stable/indexing.html

.. code:: ipython3

    ds['st']  # is the data stored, presented as a DataArray




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;st&#x27; (x: 1461, time: 3)&gt;
    array([[-8.05791e-01,  4.28741e-01, -5.13021e-01],
           [-4.58870e-01, -1.24484e-01,  9.68469e-03],
           [ 4.89174e-01, -9.57734e-02,  5.62837e-02],
           ...,
           [ 4.68457e+01,  4.72201e+01,  4.79139e+01],
           [ 3.76634e+01,  3.74649e+01,  3.83160e+01],
           [ 2.79879e+01,  2.78331e+01,  2.88055e+01]])
    Coordinates:
      * x                  (x) float64 -80.74 -80.62 -80.49 ... 104.6 104.7 104.8
        filename           (time) &lt;U31 &#x27;channel 2_20180504132202074.xml&#x27; ... &#x27;cha...
        filename_tstamp    (time) int64 20180504132202074 ... 20180504132303723
        timestart          (time) datetime64[ns] 2018-05-04T12:22:02.710000 ... 2...
        timeend            (time) datetime64[ns] 2018-05-04T12:22:32.710000 ... 2...
      * time               (time) datetime64[ns] 2018-05-04T12:22:17.710000 ... 2...
        acquisitiontimeFW  (time) timedelta64[ns] 00:00:30 00:00:30 00:00:30
    Attributes:
        name:         st
        description:  Stokes intensity
        units:        -</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'st'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 1461</li><li><span class='xr-has-index'>time</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-7ea8bdc1-cad4-488d-83d7-d3b9fac8964c' class='xr-array-in' type='checkbox' checked><label for='section-7ea8bdc1-cad4-488d-83d7-d3b9fac8964c' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>-0.8058 0.4287 -0.513 -0.4589 -0.1245 ... 38.32 27.99 27.83 28.81</span></div><div class='xr-array-data'><pre>array([[-8.05791e-01,  4.28741e-01, -5.13021e-01],
           [-4.58870e-01, -1.24484e-01,  9.68469e-03],
           [ 4.89174e-01, -9.57734e-02,  5.62837e-02],
           ...,
           [ 4.68457e+01,  4.72201e+01,  4.79139e+01],
           [ 3.76634e+01,  3.74649e+01,  3.83160e+01],
           [ 2.79879e+01,  2.78331e+01,  2.88055e+01]])</pre></div></div></li><li class='xr-section-item'><input id='section-2b72c065-eaca-49ee-9fe5-063480a5ef45' class='xr-section-summary-in' type='checkbox'  checked><label for='section-2b72c065-eaca-49ee-9fe5-063480a5ef45' class='xr-section-summary' >Coordinates: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-80.74 -80.62 ... 104.7 104.8</div><input id='attrs-0ca01c04-1179-41f0-b10e-bb3d45a13676' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-0ca01c04-1179-41f0-b10e-bb3d45a13676' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-48504ebb-8dc3-4f42-9a72-337894684be5' class='xr-var-data-in' type='checkbox'><label for='data-48504ebb-8dc3-4f42-9a72-337894684be5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>distance</dd><dt><span>description :</span></dt><dd>Length along fiber</dd><dt><span>long_description :</span></dt><dd>Starting at connector of forward channel</dd><dt><span>units :</span></dt><dd>m</dd></dl></div><div class='xr-var-data'><pre>array([-80.7443, -80.6172, -80.4901, ..., 104.567 , 104.694 , 104.821 ])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>filename</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U31</div><div class='xr-var-preview xr-preview'>&#x27;channel 2_20180504132202074.xml...</div><input id='attrs-479a1895-fcd3-42e1-a3a5-148f8c04c875' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-479a1895-fcd3-42e1-a3a5-148f8c04c875' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cbf63d7e-bf81-4e87-8d55-744afdec57d0' class='xr-var-data-in' type='checkbox'><label for='data-cbf63d7e-bf81-4e87-8d55-744afdec57d0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;channel 2_20180504132202074.xml&#x27;,
           &#x27;channel 2_20180504132232903.xml&#x27;,
           &#x27;channel 2_20180504132303723.xml&#x27;], dtype=&#x27;&lt;U31&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>filename_tstamp</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>20180504132202074 ... 2018050413...</div><input id='attrs-190f2e1b-fe41-49fa-8439-38a1f400be83' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-190f2e1b-fe41-49fa-8439-38a1f400be83' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ccc01217-65a8-467a-bd0b-5b270091d6d7' class='xr-var-data-in' type='checkbox'><label for='data-ccc01217-65a8-467a-bd0b-5b270091d6d7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([20180504132202074, 20180504132232903, 20180504132303723])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>timestart</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2018-05-04T12:22:02.710000 ... 2...</div><input id='attrs-28e629e0-197b-403e-aa86-49a196069e05' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-28e629e0-197b-403e-aa86-49a196069e05' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-05dc6ba4-e3f2-41d2-9ce8-70f155f22804' class='xr-var-data-in' type='checkbox'><label for='data-05dc6ba4-e3f2-41d2-9ce8-70f155f22804' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time start of the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2018-05-04T12:22:02.710000000&#x27;, &#x27;2018-05-04T12:22:32.702000000&#x27;,
           &#x27;2018-05-04T12:23:03.716000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>timeend</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2018-05-04T12:22:32.710000 ... 2...</div><input id='attrs-8efdc41d-481b-467d-bf29-19564f2e25e4' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-8efdc41d-481b-467d-bf29-19564f2e25e4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-88b40b02-98eb-4c07-a3b4-4fabfbdfa1d2' class='xr-var-data-in' type='checkbox'><label for='data-88b40b02-98eb-4c07-a3b4-4fabfbdfa1d2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time end of the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2018-05-04T12:22:32.710000000&#x27;, &#x27;2018-05-04T12:23:02.702000000&#x27;,
           &#x27;2018-05-04T12:23:33.716000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2018-05-04T12:22:17.710000 ... 2...</div><input id='attrs-471ceac0-1b28-4547-8100-c9c5bf1d822b' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-471ceac0-1b28-4547-8100-c9c5bf1d822b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0614e842-7c7a-4c13-9f8d-c369ef7cc99c' class='xr-var-data-in' type='checkbox'><label for='data-0614e842-7c7a-4c13-9f8d-c369ef7cc99c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time halfway the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><div class='xr-var-data'><pre>array([&#x27;2018-05-04T12:22:17.710000000&#x27;, &#x27;2018-05-04T12:22:47.702000000&#x27;,
           &#x27;2018-05-04T12:23:18.716000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>acquisitiontimeFW</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>timedelta64[ns]</div><div class='xr-var-preview xr-preview'>00:00:30 00:00:30 00:00:30</div><input id='attrs-f8943ad1-1afc-436e-abf1-fa7c7de18fe6' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-f8943ad1-1afc-436e-abf1-fa7c7de18fe6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3f9911f8-29a5-4a32-8b34-65b3c7515858' class='xr-var-data-in' type='checkbox'><label for='data-3f9911f8-29a5-4a32-8b34-65b3c7515858' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Acquisition time of the forward measurement</dd></dl></div><div class='xr-var-data'><pre>array([30000000000, 30000000000, 30000000000], dtype=&#x27;timedelta64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-20b95c3b-7de0-42ff-a69c-d040a90ee63d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-20b95c3b-7de0-42ff-a69c-d040a90ee63d' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>st</dd><dt><span>description :</span></dt><dd>Stokes intensity</dd><dt><span>units :</span></dt><dd>-</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    ds['tmp'].plot(figsize=(12, 8));

1 mean, min, max
----------------

The first argument is the dimension. The function is taken along that
dimension. ``dim`` can be any dimension (e.g., ``time``, ``x``). The
returned ``DataStore`` does not contain that dimension anymore.

Normally, you would like to keep the attributes (the informative texts
from the loaded files), so set ``keep_attrs`` to ``True``. They don’t
take any space compared to your Stokes data, so keep them.

Note that also the sections are stored as attribute. If you delete the
attributes, you would have to redefine the sections.

.. code:: ipython3

    ds_min = ds.mean(dim='time', keep_attrs=True)  # take the minimum of all data variables (e.g., Stokes, Temperature) along the time dimension

.. code:: ipython3

    ds_max = ds.max(dim='x', keep_attrs=True)  # Take the maximum of all data variables (e.g., Stokes, Temperature) along the x dimension

.. code:: ipython3

    ds_std = ds.std(dim='time', keep_attrs=True)  # Calculate the standard deviation along the time dimension

2 Selecting
-----------

What if you would like to get the maximum temperature between
:math:`x >= 20` m and :math:`x < 35` m over time? We first have to
select a section along the cable.

.. code:: ipython3

    section = slice(20., 35.)
    section_of_interest = ds.sel(x=section)

.. code:: ipython3

    section_of_interest_max = section_of_interest.max(dim='x')

What if you would like to have the measurement at approximately
:math:`x=20` m?

.. code:: ipython3

    point_of_interest = ds.sel(x=20., method='nearest')

3 Selecting by index
--------------------

What if you would like to see what the values on the first timestep are?
We can use isel (index select)

.. code:: ipython3

    section_of_interest = ds.isel(time=slice(0, 2))  # The first two time steps

.. code:: ipython3

    section_of_interest = ds.isel(x=0)

4 Downsample (time dimension)
-----------------------------

We currently have measurements at 3 time steps, with 30.001 seconds
inbetween. For our next exercise we would like to down sample the
measurements to 2 time steps with 47 seconds inbetween. The calculated
variances are not valid anymore. We use the function
``resample_datastore``.

.. code:: ipython3

    ds_resampled = ds.resample_datastore(how='mean', time="47S")

5 Upsample / Interpolation (length and time dimension)
------------------------------------------------------

So we have measurements every 0.12 cm starting at :math:`x=0` m. What if
we would like to change our coordinate system to have a value every 12
cm starting at :math:`x=0.05` m. We use (linear) interpolation,
extrapolation is not supported. The calculated variances are not valid
anymore.

.. code:: ipython3

    x_old = ds.x.data
    x_new = x_old[:-1] + 0.05 # no extrapolation
    ds_xinterped = ds.interp(coords={'x': x_new})

We can do the same in the time dimension

.. code:: ipython3

    import numpy as np
    time_old = ds.time.data
    time_new = time_old + np.timedelta64(10, 's')
    ds_tinterped = ds.interp(coords={'time': time_new})

