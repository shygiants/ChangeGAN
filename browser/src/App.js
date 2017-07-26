import React, { Component } from 'react';
import './App.css';
import './assets/react-toolbox/theme.css';
import Slider from 'react-toolbox/lib/slider/Slider';

import theme from './assets/react-toolbox/theme'
import ThemeProvider from 'react-toolbox/lib/ThemeProvider';

class App extends Component {
  constructor(props) {
    super(props)
    this.state = {
      step: 78375,
      batch: 1
    };

    this.batchSize = 50

    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(value) {
    this.setState({
      batch: value
    });
  }

  render() {
    return (
      <ThemeProvider theme={theme}>
        <section>
          <div className="slider">
            <Slider pinned snaps min={1} max={10} step={1} editable value={this.state.batch} onChange={this.handleChange} />  
          </div>
            
          {[...Array(this.batchSize).keys()].map(idx => 
              <div className="image-section">
                <img src={`https://storage.googleapis.com/mlcampjeju2017-mlengine/eval-output/${this.state.step}/inputs_a_${this.state.batch}_${idx}.jpg`} alt=""/>
                <img src={`https://storage.googleapis.com/mlcampjeju2017-mlengine/eval-output/${this.state.step}/inputs_b_${this.state.batch}_${idx}.jpg`} alt=""/>
                <img src={`https://storage.googleapis.com/mlcampjeju2017-mlengine/eval-output/${this.state.step}/outputs_ba_${this.state.batch}_${idx}.jpg`} alt=""/>
                <img src={`https://storage.googleapis.com/mlcampjeju2017-mlengine/eval-output/${this.state.step}/outputs_ab_${this.state.batch}_${idx}.jpg`} alt=""/>
              </div>
            )}
        </section>
      </ThemeProvider>
    );
  }
}

export default App;
