import React, { Component } from 'react';
import Typist from 'react-typist';
import './styles/App.css';
import io from 'socket.io-client';
import 'react-typist/dist/Typist.css';
// import imgOne from './img/1.png'
// import imgTwo from './img/2.png'
import white from './img/white.png'

const SERVER_IP = '64.62.141.30'
const PORT = '3332';
const ROUTE = '/query'
const URL = `http://${SERVER_IP}:${PORT}${ROUTE}`

class App extends Component {
  state = {
    connected: false,
    srcImgOne: null,
    srcImgTwo: null,
    latestUpdate: 1,
    textArea: '',
    socket: null,
    auto: false
  }

  componentDidMount(){    
    this.setState({
      socket: io(URL)
    }, () => {
      this.state.socket.on('connect', () => {
        this.setState({ connected: true })
      });
      this.state.socket.on('update_response', (data) => {
        if (this.state.latestUpdate === 1) {
          this.setState({
            srcImgTwo: "data:image/jpg;base64," + data.image,
            latestUpdate: 2,
          })
        } else {
          this.setState({
            srcImgOne: "data:image/jpg;base64," + data.image,
            latestUpdate: 1,
          })
        }
      });
    })
    
    if (this.state.auto) {
      setInterval(() => {
        const { socket } = this.state;
        const caption = document.getElementsByClassName('Typist')[0].innerText;
        socket.emit('update_request', {
          caption: caption
        });
      }, 3000);
    }
  }

  handleTextArea = (e) => {
    const { socket } = this.state;
    const caption = e.target.value
    
    if (caption.length % 5 === 0) {
      socket.emit('update_request', {
        caption: e.target.value
      });
    }
    this.setState({
      textArea: e.target.value
    });
  }

  render() {
    const { connected, srcImgOne, srcImgTwo, latestUpdate, textArea, auto } = this.state;

    let classImgOne = 'FadeIn';
    let classImgTwo = 'FadeOut';
    if (latestUpdate === 2) {
      classImgOne = 'FadeOut';
      classImgTwo = 'FadeIn';
    }

    return (
      <div className="App">
        <span id="Status" className={connected ? "Connected" : null}></span>
        <div className="Left">
          {
            !auto
            ?
            <div className="TextArea">
              <textarea
                placeholder='Write something...'
                cols="30" 
                rows="10"
                onChange={this.handleTextArea} 
                value={textArea}
              >
              </textarea>
            </div>
            :
            <Typist avgTypingDelay={90}>
                The machine does not work, but that, to my mind, is a secondary matter. Nor do the machines that attempt to produce continuous movement, whose plans add mystery to the pages of the most effusive encyclopedias; metaphysical or theological theories do not work either (....), but their well-known and famous uselessness does not reduce their interest
            </Typist>
          }
        </div>
        <div className="Right">
          <img className={classImgOne} src={srcImgOne ? srcImgOne : white} alt='img one'/>
          <img className={classImgTwo} id='ImgTwo' src={srcImgTwo ? srcImgTwo : white} alt='img two'/>
        </div>
        <div className='Credits'>
          <p>Made by <a href="https://cvalenzuelab.com/">Cris Valenzuela</a> with <a href="https://runwayml.com/">Runway</a> | Using <a href="https://github.com/taoxugit/AttnGAN/">AttnGAN</a> | GPU hosting thanks to <a href="https://www.paperspace.com/">Paperspace</a>  </p>
        </div>
      </div>
    );
  }
}

export default App;
